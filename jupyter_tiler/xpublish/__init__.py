from functools import cache
from typing import Any

from xarray import DataArray

from jupyter_tiler.xpublish._server import XpublishServer


@cache
def _get_server() -> XpublishServer:
    return XpublishServer()


async def add_data_array(
    data_array: DataArray,
    *,
    colormap_range: tuple[float, float] | None = None,
    # ...,
    **kwargs: str | int,
) -> str:
    """Adds a DataArray to the xpublish-tiles server.

    The xpublish-tiles server is lazily started when the first DataArray is added.

    Args:
        data_array: An Xarray DataArray to dynamically tile for visualization.
        colormap_range: The range of data values ``(min, max)`` to be colormapped
        kwargs: Additional query parameters to include in the TiTiler request URL.

    Returns:
        A URL template pointing to the new tile endpoint.
    """
    return await _get_server().add_data_array(
        data_array,
        colormap_range=colormap_range,
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
