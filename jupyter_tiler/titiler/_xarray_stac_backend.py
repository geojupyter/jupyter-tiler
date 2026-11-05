import json
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Any, Literal, cast

import attr
import dask
import numpy as np
import pystac
import stackstac
import xarray
from attrs import define
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
from fastapi import Depends, Path, Query
from geojson_pydantic.geometries import Geometry
from morecantile import Tile, TileMatrixSets
from morecantile import tms as morecantile_tms
from pydantic import Field
from pystac_client import ItemSearch
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling
from rio_tiler.errors import NoAssetFoundError
from rio_tiler.models import ImageData
from rio_tiler.types import ColorMapType
from rio_tiler.utils import Timer
from starlette.responses import Response
from titiler.core.dependencies import (
    ColorMapParams,
    DefaultDependency,
    ImageRenderingParams,
)
from titiler.core.factory import BaseFactory, img_endpoint_params
from titiler.core.resources.enums import ImageType
from titiler.core.utils import render_image
from titiler.stacapi.backend import STACAPIBackend
from titiler.stacapi.dependencies import (
    APIParams,
    CollectionSearch,
    Search,
    STACAPIExtensionParams,
)
from titiler.stacapi.settings import CacheSettings, ItemsSettings, RetrySettings
from urllib3 import Retry
from xarray.core.types import InterpOptions

cache_config = CacheSettings()
retry_config = RetrySettings()
items_config = ItemsSettings(max_items=8)

ttl_cache: Any = TTLCache(maxsize=cache_config.maxsize, ttl=cache_config.ttl)
_DEFAULT_DASK_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _slice_to_bounds(
    data_array: xarray.DataArray,
    bounds: tuple[float, float, float, float],
    *,
    x_name: str = "x",
    y_name: str = "y",
) -> xarray.DataArray:
    """Slice a DataArray to bounds while handling ascending or descending axes."""
    if x_name not in data_array.coords or y_name not in data_array.coords:
        return data_array

    west, south, east, north = bounds

    x_values = data_array.coords[x_name].values
    y_values = data_array.coords[y_name].values

    x_slice = slice(west, east) if x_values[0] <= x_values[-1] else slice(east, west)

    if y_values[0] <= y_values[-1]:
        y_slice = slice(south, north)
    else:
        y_slice = slice(north, south)

    return data_array.sel({x_name: x_slice, y_name: y_slice})


def _filter_items_by_nodata(
    data_array: xarray.DataArray,
    max_nodata_ratio: float,
) -> tuple[xarray.DataArray, np.ndarray]:
    """Drop STAC items (``time`` slices) with too much nodata in the tile.

    Each ``time`` slice of a stackstac result comes from one STAC item. The
    nodata ratio of an item is the fraction of null pixels over all its
    non-``time`` dimensions (bands, y, x) within the tile.

    Returns the filtered DataArray and a boolean mask of the kept items,
    aligned with the ``time`` dimension.
    """
    if "time" not in data_array.dims:
        return data_array, np.ones(1, dtype=bool)

    reduce_dims = [dim for dim in data_array.dims if dim != "time"]
    nodata_ratio = data_array.isnull().mean(dim=reduce_dims).compute()
    keep = (nodata_ratio <= max_nodata_ratio).values
    return data_array.isel(time=np.flatnonzero(keep)), keep


def _nodata_alpha_mask(
    data_array: xarray.DataArray,
    *,
    x_name: str = "x",
    y_name: str = "y",
) -> np.ndarray:
    """Build a uint8 alpha mask (255 = valid, 0 = nodata) from a DataArray.

    A pixel is nodata when all values across the non-spatial dimensions
    (e.g. ``time`` and ``band``) are null.
    """
    reduce_dims = [dim for dim in data_array.dims if dim not in (x_name, y_name)]
    valid = ~data_array.isnull().all(dim=reduce_dims)
    return (valid.compute().values.astype("uint8")) * 255


def _apply_nodata_transparency(
    image: ImageData,
    data_array: xarray.DataArray,
) -> ImageData:
    """Render pixels that are nodata in the source DataArray as transparent.

    The mask is combined with any alpha mask already set by ``array_to_image``:
    a pixel stays transparent if either mask marks it as nodata. If the image
    grid does not match the DataArray grid (e.g. ``array_to_image`` regridded),
    the image is returned unchanged.
    """
    alpha = _nodata_alpha_mask(data_array)
    if alpha.shape != (image.height, image.width):
        return image
    if image.alpha_mask is not None:
        alpha = np.minimum(image.alpha_mask, alpha)
    image.alpha_mask = alpha
    return image


def _resample_dataarray_to_viewport(  # noqa: PLR0913
    data_array: xarray.DataArray,
    *,
    x_name: str,
    y_name: str,
    viewport_width: int,
    viewport_height: int,
    method: InterpOptions = "linear",
) -> xarray.DataArray:
    """Resample a DataArray to an approximate viewport-sized grid."""
    if viewport_width <= 0 or viewport_height <= 0:
        return data_array

    if x_name not in data_array.coords or y_name not in data_array.coords:
        return data_array

    x_values = data_array.coords[x_name].values
    y_values = data_array.coords[y_name].values
    if len(x_values) < 2 or len(y_values) < 2:
        return data_array

    x_target = np.linspace(float(x_values[0]), float(x_values[-1]), viewport_width)
    y_target = np.linspace(float(y_values[0]), float(y_values[-1]), viewport_height)
    return data_array.interp({x_name: x_target, y_name: y_target}, method=method)


@dataclass(init=False)
class BackendParams(DefaultDependency):
    """backend parameters."""

    api_params: APIParams = field(init=False)

    def __init__(
        self,
        url: str,
    ) -> None:
        """Initialize BackendParams

        The STAC API URL is provided by the tiler factory configuration rather
        than app state or request query params.
        """
        self.api_params = APIParams(url=url)


@attr.s
class XarraySTACAPIBackend(STACAPIBackend):
    """STACAPI Mosaic Backend which provides tiles as xarrays."""

    # in STACAPI backend assets are STAC Items as dict
    def asset_name(self, item: pystac.Item) -> str:  # type: ignore[override]
        """Get asset name."""
        return f"{item.collection_id}/{item.id}"

    # NOTE: Custom get_assets method which return `pystac.Item` instead of `dict`
    @cached(
        ttl_cache,
        key=lambda self, geom, **kwargs: hashkey(
            self.api_params["url"],
            str(geom),
            json.dumps(self.input),
            json.dumps(self.api_params.get("headers", {})),
            json.dumps(kwargs, sort_keys=True, default=str),
        ),
        lock=Lock(),
    )
    def get_assets(  # type: ignore[override]
        self,
        geom: Geometry,
        sortby: list[dict[str, Any]] | None = None,
        limit: int | None = None,
        max_items: int | None = None,
        stac_search_kwargs: str = "{}",
    ) -> list[pystac.Item]:
        """Find assets."""

        search_query = {
            **self.input,
            "method": "GET" if self.input.get("filter") else "POST",
            "sortby": sortby,
            "limit": limit or items_config.items_per_page,
            "max_items": max_items or items_config.max_items,
        }

        stac_api_io = StacApiIO(
            max_retries=Retry(
                total=retry_config.retry,
                backoff_factor=retry_config.retry_factor,
            ),
            headers=self.api_params.get("headers", {}),
        )

        params = {
            **search_query,
            "intersects": geom.model_dump(exclude_none=True),
            **json.loads(stac_search_kwargs),
        }
        params.pop("bbox", None)

        results = ItemSearch(
            f"{self.api_params['url']}/search", stac_io=stac_api_io, **params
        )
        return list(results.items())

    def tile(  # type: ignore[override]  # noqa: PLR0913
        self,
        *,
        x: int,
        y: int,
        z: int,
        assets: list[str] | None = None,
        search_options: dict[str, Any] | None = None,
        tilesize: int | None = None,
        max_items: int = 4,
        resolution_scale: float = 1.0,
        resampling: Resampling = Resampling.nearest,
        viewport_width: int = 0,
        viewport_height: int = 0,
        viewport_resampling: InterpOptions = "linear",
        stac_search_kwargs: dict[str, Any] | None = None,
        max_nodata_ratio: float | None = None,
    ) -> tuple[xarray.DataArray, list[str]]:
        """Get Tile from multiple assets."""
        timings = []
        with Timer() as t:
            search_options = {
                **(search_options or {}),
                "limit": max_items,
                "max_items": max_items,
                "stac_search_kwargs": json.dumps(OrderedDict(stac_search_kwargs or {})),
            }

            # NOTE: This might raise typing issue because in the original backend
            # `assets_for_tile`
            # return a list of dict, but in our custom backend it return a list
            # of `pystac.Item`
            mosaic_assets = cast(
                "list[pystac.Item]",
                self.assets_for_tile(x, y, z, **search_options),
            )
        timings.append(("search", round(t.elapsed * 1000, 2)))

        if not mosaic_assets:
            raise NoAssetFoundError(f"No assets found for tile {z}-{x}-{y}")

        matrix = self.tms.matrix(z)
        height = tilesize or matrix.tileHeight
        width = tilesize or matrix.tileWidth

        bounds = self.tms.xy_bounds(Tile(x, y, z))
        x_res = (bounds[2] - bounds[0]) / width
        y_res = (bounds[3] - bounds[1]) / height
        effective_resolution = (
            x_res * resolution_scale,
            y_res * resolution_scale,
        )

        # Create Xarray DataArray from stac items
        plain_items = [item.to_dict(transform_hrefs=False) for item in mosaic_assets]
        xr_stack = stackstac.stack(
            plain_items,
            assets=assets,
            epsg=self.tms.crs.to_epsg(),
            bounds=bounds,
            resolution=effective_resolution,
            resampling=resampling,
        )
        xr_stack = _slice_to_bounds(xr_stack, bounds)

        if max_nodata_ratio is not None:
            xr_stack, keep = _filter_items_by_nodata(xr_stack, max_nodata_ratio)
            mosaic_assets = [
                item for item, kept in zip(mosaic_assets, keep, strict=True) if kept
            ]
            if not mosaic_assets:
                raise NoAssetFoundError(
                    f"All assets for tile {z}-{x}-{y} have more than "
                    f"{max_nodata_ratio:.0%} nodata"
                )

        xr_stack = _resample_dataarray_to_viewport(
            xr_stack,
            x_name="x",
            y_name="y",
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            method=viewport_resampling,
        )

        asset_used = [self.asset_name(asset) for asset in mosaic_assets]

        return xr_stack, asset_used


@define(kw_only=True)
class XarraySTACTilerFactory(BaseFactory):
    """TiTiler factory serving tiles computed from STAC items as xarrays.

    The STAC API URL (``stac_url``) is a property of the factory: every route
    registered by this factory queries that same STAC API, so adding several
    sources to the same app never mixes STAC APIs.

    ``viewport_width``/``viewport_height`` exist because stackstac must pick a
    single output resolution for the stacked array, defaulting to the finest
    source resolution, which can be much finer than what is visible on screen.
    Downsampling the stack to roughly the viewport size avoids computing pixels
    that are never displayed. The viewport can change dynamically (pan, zoom,
    resize), so ``0`` disables this and keeps the native tile resolution.
    """

    backend: type[XarraySTACAPIBackend] = XarraySTACAPIBackend
    backend_dependency: type[BackendParams] = BackendParams

    search_dependency: Callable[..., Search] = CollectionSearch
    assets_accessor_dependency: type[DefaultDependency] = STACAPIExtensionParams

    colormap_dependency: Callable[..., ColorMapType | None] = ColorMapParams
    render_dependency: type[DefaultDependency] = ImageRenderingParams

    supported_tms: TileMatrixSets = morecantile_tms

    render_func: Callable[..., tuple[bytes, str]] = render_image

    array_to_image: Callable[[xarray.DataArray], ImageData]

    stac_url: str

    # If None, all assets of each item are used (only selection by asset ID
    # is supported, not by mimetype).
    assets: list[str] | None = None
    max_items: int = 4
    resolution_scale: float = 1.0
    resampling: Resampling = Resampling.nearest
    viewport_width: int = 0
    viewport_height: int = 0
    viewport_resampling: InterpOptions = "linear"
    stac_search_kwargs: dict[str, Any] | None = None
    # If set, STAC items whose nodata fraction within a tile exceeds this
    # ratio are dropped before rendering. None disables the filtering.
    max_nodata_ratio: float | None = None
    # If True (default), pixels that are nodata in the stacked DataArray are
    # rendered transparent (alpha 0) in the tile image.
    nodata_transparent: bool = True

    def register_routes(self) -> None:
        self.tile()

    ############################################################################
    # /tiles
    ############################################################################
    def tile(self) -> None:
        """Register /tiles endpoint."""

        @self.router.get(
            "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}",
            operation_id=f"{self.operation_prefix}getTile",
            **img_endpoint_params,
        )
        @self.router.get(
            "/collections/{collection_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}",
            operation_id=f"{self.operation_prefix}getTileWithFormat",
            **img_endpoint_params,
        )
        def tile(  # noqa: PLR0913
            z: Annotated[
                int,
                Path(
                    description="""
                    Identifier (Z) selecting one of the scales defined
                    in the TileMatrixSet and representing the scaleDenominator the tile.
                    """,
                ),
            ],
            x: Annotated[
                int,
                Path(
                    description="""
                    Column (X) index of the tile on the selected TileMatrix.
                    It cannot exceed the MatrixHeight-1 for the selected TileMatrix.
                    """,
                ),
            ],
            y: Annotated[
                int,
                Path(
                    description="""
                    Row (Y) index of the tile on the selected TileMatrix.
                    It cannot exceed the MatrixWidth-1 for the selected TileMatrix.
                    """,
                ),
            ],
            tileMatrixSetId: Annotated[  # noqa: N803
                Literal["WebMercatorQuad"],
                Path(
                    description="""
                    Identifier selecting one of the TileMatrixSetId supported.
                    """
                ),
            ],
            format: Annotated[  # noqa: A002
                ImageType | None,
                Field(
                    description="""
                    Default will be automatically defined if the output image
                    needs a mask (png) or not (jpeg).
                    """
                ),
            ] = None,
            tilesize: Annotated[
                int | None,
                Query(gt=0, description="Tilesize in pixels."),
            ] = None,
            search: Search = Depends(self.search_dependency),  # noqa: B008
            backend_params: BackendParams = Depends(  # noqa: B008
                lambda: BackendParams(url=self.stac_url)
            ),
            assets_accessor_params: STACAPIExtensionParams = Depends(  # noqa: B008
                self.assets_accessor_dependency
            ),
            render_params: ImageRenderingParams = Depends(self.render_dependency),  # noqa: B008
        ) -> Response:
            """Create map tile from a dataset."""
            tms = self.supported_tms.get(tileMatrixSetId)

            with self.backend(
                search,
                tms=tms,
                **backend_params.as_dict(),
            ) as src_dst:
                tile = src_dst.tile(
                    x=x,
                    y=y,
                    z=z,
                    assets=self.assets,
                    tilesize=tilesize,
                    search_options=assets_accessor_params.as_dict(),
                    max_items=self.max_items,
                    resolution_scale=self.resolution_scale,
                    resampling=self.resampling,
                    viewport_width=self.viewport_width,
                    viewport_height=self.viewport_height,
                    viewport_resampling=self.viewport_resampling,
                    stac_search_kwargs=self.stac_search_kwargs,
                    max_nodata_ratio=self.max_nodata_ratio,
                )

            with dask.config.set(
                scheduler="threads",
                num_workers=_DEFAULT_DASK_WORKERS,
            ):
                image = self.array_to_image(tile[0])
                if self.nodata_transparent:
                    image = _apply_nodata_transparency(image, tile[0])

            content, media_type = self.render_func(
                image,
                output_format=format,
                **render_params.as_dict(),
            )

            return Response(content, media_type=media_type)
