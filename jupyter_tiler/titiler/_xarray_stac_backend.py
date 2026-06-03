import json
from collections.abc import Callable
from threading import Lock
from typing import Annotated, Any, Literal, cast

import attr
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
    BackendParams,
    CollectionSearch,
    Search,
    STACAPIExtensionParams,
)
from titiler.stacapi.settings import CacheSettings, ItemsSettings, RetrySettings
from urllib3 import Retry

cache_config = CacheSettings()
retry_config = RetrySettings()
items_config = ItemsSettings()

ttl_cache = TTLCache(maxsize=cache_config.maxsize, ttl=cache_config.ttl)


@attr.s
class XarraySTACAPIBackend(STACAPIBackend):
    """STACAPI Mosaic Backend which provides tiles as xarrays."""

    # in STACAPI backend assets are STAC Items as dict
    def asset_name(self, item: pystac.Item) -> str:  # type: ignore[override]
        """Get asset name."""
        return f"{item.collection_id}/{item.id}"

    # NOTE: Custom get_assets method which return `pystac.Item` instead of `dict`
    @cached(  # type: ignore
        ttl_cache,
        key=lambda self, geom, **kwargs: hashkey(
            self.api_params["url"],
            str(geom),
            json.dumps(self.input),
            json.dumps(self.api_params.get("headers", {})),
            **kwargs,
        ),
        lock=Lock(),
    )
    def get_assets(  # type: ignore[override]
        self,
        geom: Geometry,
        sortby: list[dict] | None = None,
        limit: int | None = None,
        max_items: int | None = None,
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
            "intersects": geom.model_dump_json(exclude_none=True),
        }
        params.pop("bbox", None)

        results = ItemSearch(
            f"{self.api_params['url']}/search", stac_io=stac_api_io, **params
        )
        return list(results.items())

    def tile(  # type: ignore[override]
        self,
        x: int,
        y: int,
        z: int,
        search_options: dict | None = None,
        tilesize: int | None = None,
        **kwargs: Any,
    ) -> tuple[xarray.DataArray, list[str]]:
        """Get Tile from multiple assets."""
        timings = []
        with Timer() as t:
            search_options = search_options or {}

            # NOTE: This might raise typing issue because in the original backend `assets_for_tile`
            # return a list of dict, but in our custom backend it return a list of `pystac.Item`
            mosaic_assets = cast(
                "list[pystac.Item]",  # type: ignore
                self.assets_for_tile(x, y, z, **search_options),  # type: ignore
            )
        timings.append(("search", round(t.elapsed * 1000, 2)))

        if not mosaic_assets:
            raise NoAssetFoundError(f"No assets found for tile {z}-{x}-{y}")

        matrix = self.tms.matrix(z)
        height = tilesize or matrix.tileHeight
        width = tilesize or matrix.tileWidth

        bounds = self.tms.bounds(Tile(x, y, z))
        x_res = (bounds[2] - bounds[0]) / width
        y_res = (bounds[3] - bounds[1]) / height

        # Create Xarray DataArray from stac items
        xr_stack = stackstac.stack(
            mosaic_assets,
            epsg=self.tms.crs.to_epsg(),
            bounds=self.tms.bounds(Tile(x, y, z)),
            resolution=(x_res, y_res),
        )

        asset_used = [self.asset_name(asset) for asset in mosaic_assets]

        return xr_stack, asset_used


@define(kw_only=True)
class XarraySTACTilerFactory(BaseFactory):
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

    def register_routes(self):
        self.tile()

    ############################################################################
    # /tiles
    ############################################################################
    def tile(self):
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
        def tile(
            z: Annotated[
                int,
                Path(
                    description="Identifier (Z) selecting one of the scales defined in the TileMatrixSet and representing the scaleDenominator the tile.",
                ),
            ],
            x: Annotated[
                int,
                Path(
                    description="Column (X) index of the tile on the seleced TileMatrix. It cannot exceed the MatrixHeight-1 for the selected TileMatrix.",
                ),
            ],
            y: Annotated[
                int,
                Path(
                    description="Row (Y) index of the tile on the selected TileMatrix. It cannot exceed the MatrixWidth-1 for the selected TileMatrix.",
                ),
            ],
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                Path(
                    description="Identifier selecting one of the TileMatrixSetId supported."
                ),
            ],
            format: Annotated[
                ImageType | None,
                Field(
                    description="Default will be automatically defined if the output image needs a mask (png) or not (jpeg)."
                ),
            ] = None,
            tilesize: Annotated[
                int | None,
                Query(gt=0, description="Tilesize in pixels."),
            ] = None,
            search=Depends(self.search_dependency),
            backend_params=Depends(self.backend_dependency),
            assets_accessor_params=Depends(self.assets_accessor_dependency),
            # colormap=Depends(self.colormap_dependency),
            render_params=Depends(self.render_dependency),
        ):
            """Create map tile from a dataset."""
            tms = self.supported_tms.get(tileMatrixSetId)

            backend_kwargs = backend_params.as_dict()
            backend_kwargs.setdefault("api_params", {})
            backend_kwargs["api_params"]["url"] = self.stac_url

            with self.backend(
                search,
                tms=tms,
                **backend_params.as_dict(),
            ) as src_dst:
                xr = src_dst.tile(
                    x,
                    y,
                    z,
                    tilesize=tilesize,
                    search_options=assets_accessor_params.as_dict(),
                )

            image = self.array_to_image(xr)

            content, media_type = self.render_func(
                image,
                output_format=format,
                # colormap=colormap,
                **render_params.as_dict(),
            )

            return Response(content, media_type=media_type)
