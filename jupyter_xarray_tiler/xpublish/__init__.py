from functools import cache
from typing import Any

from xarray import DataArray

from jupyter_xarray_tiler.xpublish._server import XpublishServer


@cache
def _get_server() -> XpublishServer:
    return XpublishServer()


async def add_data_array(
    data_array: DataArray,
    *,
    rescale: tuple[float, float] | None = None,
    # ...,
    **kwargs: str | int,
) -> str:
    """Adds a DataArray to the xpublish-tiles server and returns a URL template.

    The xpublish-tiles server is lazily started when the first DataArray is added.

    Args:
        rescale: Comma (',') delimited Min,Max range.
        kwargs: Additional query parameters to include in the TiTiler request URL.

    Returns:
        A URL pointing to the new tile endpoint.
    """
    return await _get_server().add_data_array(
        data_array,
        rescale=rescale,
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
