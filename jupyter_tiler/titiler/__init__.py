from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from rio_tiler.models import ImageData
from titiler.core.algorithm.base import BaseAlgorithm
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer


@cache
def _get_server() -> TiTilerServer:
    return TiTilerServer()


async def add_data_array(
    data_array: DataArray,
    *,
    colormap_name: str = "viridis",
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
        colormap_range=colormap_range,
        tile_dim_scale=tile_dim_scale,
        algorithm=algorithm,
        **kwargs,
    )


def transparent_image(width: int = 256, height: int = 256) -> ImageData:
    arr = np.ma.masked_all((1, height, width), dtype=np.uint8)

    return ImageData(
        arr,
        assets=None,
        crs=None,
        bounds=None,
    )


def default_array_to_image(xr: DataArray) -> ImageData:
    """
    Convert a stackstac xarray.DataArray into rio-tiler ImageData.

    Output:
        ImageData usable by titiler render functions.
    """
    height = xr.sizes.get("y", 256)
    width = xr.sizes.get("x", 256)

    if "time" in xr.dims and xr.sizes["time"] == 0:
        return transparent_image(width, height)

    if "band" in xr.dims and xr.sizes["band"] == 0:
        return transparent_image(width, height)

    data = xr.isel(time=0) if "time" in xr.dims else xr
    if "band" not in data.dims:
        data = data.expand_dims(dim="band", axis=0)

    arr = data.data

    if hasattr(arr, "compute"):
        arr = arr.compute()

    if not np.ma.isMaskedArray(arr):
        arr = np.ma.masked_invalid(arr)

    if arr.size == 0 or np.ma.count(arr) == 0:
        return transparent_image(width, height)

    if arr.shape[0] >= 3:
        arr = arr[:3, :, :]
    elif arr.shape[0] == 2:
        arr = np.ma.concatenate((arr, arr[1:2, :, :]), axis=0)

    if arr.dtype != np.uint8:
        scaled = np.ma.empty_like(arr, dtype=np.float32)
        for band_idx in range(arr.shape[0]):
            band = arr[band_idx]
            if np.ma.count(band) == 0:
                scaled[band_idx] = np.ma.masked_all(band.shape, dtype=np.float32)
                continue

            band_values = band.compressed()
            band_min = float(np.nanpercentile(band_values, 2))
            band_max = float(np.nanpercentile(band_values, 98))
            if np.isclose(band_max, band_min):
                band_max = band_min + 1.0

            scaled_band = (band - band_min) / (band_max - band_min)
            scaled[band_idx] = np.ma.clip(scaled_band, 0.0, 1.0)

        arr = (scaled * 255).astype(np.uint8)

    return ImageData(
        arr,
        assets=None,
        crs=str(data.rio.crs) if hasattr(data, "rio") else None,
        bounds=data.rio.bounds() if hasattr(data, "rio") else None,
    )


async def add_stac_array(
    stac_url: str,
    collection_id: str,
    array_to_image: Callable[[DataArray], ImageData] | None = None,
    *,
    assets: list[str] | None = None,
    max_items: int = 4,
    resolution_scale: float = 2.0,
    resampling: str = "nearest",
    viewport_width: int = 0,
    viewport_height: int = 0,
    viewport_resampling: str = "linear",
    **kwargs: str | int,
) -> str:
    """Add a STAC API source to the TiTiler server.

    The default ``array_to_image`` works for most multi-band and single-band stacks:
    it computes the first time slice, masks invalid values, applies robust percentile
    scaling, and returns an ``ImageData`` tile.

    Args:
        stac_url: Root STAC API URL.
        array_to_image: Optional callable converting a stackstac ``DataArray`` to
            ``ImageData``. If omitted, :func:`default_array_to_image` is used.
        collection_id: STAC collection ID used in the tile URL path.
        assets: Optional STAC asset names passed to stackstac.
        max_items: Max number of STAC items to combine per tile. Lower is faster.
        resolution_scale: Multiplier applied to stackstac output resolution.
            Values greater than ``1`` reduce detail and improve performance.
        resampling: stackstac resampling method, e.g. ``nearest`` or ``bilinear``.
        viewport_width: Optional target viewport width in pixels for post-stack
            downsampling. ``0`` disables viewport resampling.
        viewport_height: Optional target viewport height in pixels for post-stack
            downsampling. ``0`` disables viewport resampling.
        viewport_resampling: Interpolation method for viewport downsampling.
            Typical values are ``linear`` or ``nearest``.
        kwargs: Extra query parameters appended to the tile URL.

    Returns:
        A URL template pointing to the new STAC tile endpoint.

    Example:
        NDWI process function for use with a JupyterGIS-style API:

        .. code-block:: python

            import numpy as np
            from rio_tiler.models import ImageData

            def ndwi_process(stack):
                height = stack.sizes.get("y", 256)
                width = stack.sizes.get("x", 256)

                if "time" in stack.dims and stack.sizes["time"] == 0:
                    return ImageData(np.ma.masked_all((1, height, width), dtype=np.uint8))

                # stackstac band coordinates can be string labels ("B03", "B08")
                # or numeric values (1, 2, ...) depending on source metadata.
                data = stack.isel(time=0) if "time" in stack.dims else stack

                if "band" not in data.dims or data.sizes["band"] < 2:
                    return ImageData(np.ma.masked_all((1, height, width), dtype=np.uint8))

                try:
                    green = data.sel(band="B03").data.astype(np.float32)
                    nir = data.sel(band="B08").data.astype(np.float32)
                except Exception:
                    # Fallback for unlabeled/numeric band coords.
                    # Assumes assets=["B03", "B08"] order.
                    green = data.isel(band=0).data.astype(np.float32)
                    nir = data.isel(band=1).data.astype(np.float32)

                ndwi = (green - nir) / (green + nir + 1e-6)
                ndwi = np.ma.masked_invalid(ndwi)
                ndwi_01 = np.ma.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
                return ImageData((ndwi_01[np.newaxis, :, :] * 255).astype(np.uint8))

            # User-facing integration in a map widget:
            # doc.add_stac_array_layer(stac_url, ndwi_process)
    """
    if array_to_image is None:
        array_to_image = default_array_to_image

    return await _get_server().add_stac_array(
        stac_url=stac_url,
        collection_id=collection_id,
        assets=assets,
        max_items=max_items,
        resolution_scale=resolution_scale,
        resampling=resampling,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        viewport_resampling=viewport_resampling,
        array_to_image=array_to_image,
        **kwargs,
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
