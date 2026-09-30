from functools import cache
from typing import Any

from rio_tiler.types import ColorMapType
from titiler.core.algorithm.base import BaseAlgorithm
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer


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
