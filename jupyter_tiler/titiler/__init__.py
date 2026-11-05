from collections.abc import Callable
from functools import cache
from typing import Any

from rasterio.enums import Resampling
from rio_tiler.models import ImageData
from rio_tiler.types import ColorMapType
from titiler.core.algorithm.base import BaseAlgorithm
from xarray import DataArray
from xarray.core.types import InterpOptions

from jupyter_tiler.titiler._server import ResamplingMethod, TiTilerServer


@cache
def _get_server() -> TiTilerServer:
    return TiTilerServer()


async def add_data_array(  # noqa: PLR0913
    data_array: DataArray,
    *,
    colormap_name: str | None = None,
    colormap: ColorMapType | None = None,
    colormap_range: tuple[float, float] | None = None,
    tile_dim_scale: int = 1,
    algorithm: BaseAlgorithm | None = None,
    **kwargs: str | int,
) -> str:
    """Adds a DataArray to the TiTiler server.

    The TiTiler server is lazily started when the first DataArray is added.

    Args:
        data_array: An Xarray DataArray to dynamically tile for visualization.
        colormap_name: A ``rio-tiler``-supported colormap name.
            See the `rio-tiler docs <https://cogeotiff.github.io/rio-tiler/latest/api/rio_tiler/colormap/#rio_tiler.colormap.ColorMaps.list>`_
            for details.
            Defaults to "viridis".
            Mutually exclusive with ``colormap``.
        colormap: A custom colormap to apply to the dataset.
            One of:

            * A dict of ``{0..255: (r, g, b, a)}`` for a continuous colormap.
            * A sparse dict of ``{exact_value: (r, g, b, a)}`` for a discrete colormap.
            * A list of ``[((min, max), (r, g, b, a)), ((min2, max2), (r, g, b, a))]``
              for an interval colormap.

            Mutually exclusive with ``colormap_name``.
        colormap_range: The range of data values ``(min, max)`` to be colormapped
        tile_dim_scale: Tile size scale. Default ``1`` corresponds to 256*256px tiles.
        algorithm: A TiTiler algorithm class.
            See the `TiTiler algorithm docs <https://developmentseed.org/titiler/examples/notebooks/Working_with_Algorithm>`_
            for details.
        kwargs: Additional query parameters to include in the TiTiler request URL.

    Returns:
        A URL template pointing to the new tile endpoint.
    """
    return await _get_server().add_data_array(
        data_array,
        colormap_name=colormap_name,
        colormap=colormap,
        colormap_range=colormap_range,
        tile_dim_scale=tile_dim_scale,
        algorithm=algorithm,
        **kwargs,
    )


async def add_stac_array(  # noqa: PLR0913
    stac_url: str,
    *,
    collection_id: str,
    array_to_image: Callable[[DataArray], ImageData],
    assets: list[str] | None = None,
    max_items: int = 4,
    resolution_scale: float = 1.0,
    resampling: ResamplingMethod | Resampling = "nearest",
    viewport_width: int = 0,
    viewport_height: int = 0,
    viewport_resampling: InterpOptions = "linear",
    stac_search_kwargs: dict[str, Any] | None = None,
    max_nodata_ratio: float | None = None,
) -> str:
    """Add a STAC API source to the TiTiler server.

    This registers a new tile endpoint backed by a STAC API search and a user-provided
    ``array_to_image`` callable which converts the stacked ``DataArray``
    into ``ImageData``.

    Args:
        stac_url: Root STAC API URL.
        collection_id: STAC collection ID used in the tile URL path.
        array_to_image: Callable converting a stackstac ``DataArray`` to ``ImageData``.
        assets: Optional STAC asset names passed to stackstac. Only selection by
            asset ID is supported. If ``None``, all assets of each item are used.
        max_items: Max number of STAC items to combine per tile. Lower is faster.
        resolution_scale: Multiplier applied to stackstac output resolution.
            Values greater than ``1`` produce a lower resolution output, reducing
            detail and improving performance.
        resampling: Resampling method applied by stackstac when reprojecting the
            items onto the tile grid. One of the ``rasterio.enums.Resampling``
            member names, e.g. ``nearest`` or ``bilinear``.
        viewport_width: Optional target viewport width in pixels for post-stack
            downsampling. ``0`` disables viewport resampling.
        viewport_height: Optional target viewport height in pixels for post-stack
            downsampling. ``0`` disables viewport resampling.
        viewport_resampling: Interpolation method for viewport downsampling.
            Typical values are ``linear`` or ``nearest``.
        stac_search_kwargs: Extra STAC API search parameters applied to every tile
            request, e.g. ``{"datetime": "2024-01-01/2024-12-31", "query": {...}}``.
            See the `ItemSearch API from pystac-client
            <https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.ItemSearch>`_
            for details.
        max_nodata_ratio: If set, STAC items whose nodata fraction within a tile
            exceeds this ratio (between 0 and 1) are dropped before rendering,
            e.g. ``0.9`` drops items that are more than 90% nodata in the tile.
            ``None`` disables the filtering.

    Returns:
        A URL template pointing to the new STAC tile endpoint.

    Note:
        ``viewport_width``/``viewport_height`` exist because stackstac must pick a
        single output resolution for the stacked array, defaulting to the finest
        source resolution which can be much finer than what is visible on screen.
        Downsampling the stack to roughly the viewport size avoids computing (and
        transferring) pixels that are never displayed. Since the viewport can
        change dynamically (pan/zoom/resize), these are per-source defaults; pass
        ``0`` to disable and keep the native tile resolution.

    Example:
        NDWI process function:

        .. code-block:: python

            import numpy as np
            from rio_tiler.models import ImageData

            def ndwi_process(data):
                if "time" in data.dims:
                    # pick one scene OR use median over time; choose one
                    data = data.isel(time=0)
                    # data = data.median(dim="time", skipna=True)

                green = data.sel(band="green").data.astype(np.float32)
                nir = data.sel(band="nir").data.astype(np.float32)

                ndwi = (green - nir) / (green + nir + 1e-6)
                ndwi = np.asarray(ndwi)  # drop masked-array dimensional surprises
                ndwi = np.nan_to_num(ndwi, nan=-1.0, posinf=1.0, neginf=-1.0)

                ndwi_01 = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
                pixels = (ndwi_01 * 255).astype(np.uint8)  # 2D
                return ImageData(pixels[np.newaxis, :, :])  # 3D: 1, y, x

            raster_url = await add_stac_array(
                "https://earth-search.aws.element84.com/v1",
                collection_id="sentinel-2-l2a",
                array_to_image=ndwi_process,
                assets=["green", "nir"],
                stac_search_kwargs={
                    "datetime": "2024-06-01/2024-09-30",
                    "query": {"eo:cloud_cover": {"lt": 20}},
                },
            )
    """
    return await _get_server().add_stac_array(
        stac_url=stac_url,
        collection_id=collection_id,
        array_to_image=array_to_image,
        assets=assets,
        max_items=max_items,
        resolution_scale=resolution_scale,
        resampling=resampling,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        viewport_resampling=viewport_resampling,
        stac_search_kwargs=stac_search_kwargs,
        max_nodata_ratio=max_nodata_ratio,
    )


def get_routes() -> list[dict[str, Any]]:
    """Display a list of all available routes on the TiTiler server.

    Returns:
        A list containing one dictionary per route exposed by the TiTiler server.

    Raises:
        RuntimeError: If called before the server is started.
            Always ``await`` :func:`add_data_array` first.
    """
    try:
        return _get_server().routes
    except RuntimeError as e:
        raise RuntimeError(
            "Server not started. Please `await add_data_array(...)` first."
        ) from e
