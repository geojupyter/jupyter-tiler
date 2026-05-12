from functools import cache
from typing import Any

from titiler.core.algorithm.base import BaseAlgorithm
from xarray import DataArray

from jupyter_xarray_tiler.titiler._server import TiTilerServer


@cache
def _get_server() -> TiTilerServer:
    return TiTilerServer()


async def add_data_array(
    data_array: DataArray,
    *,
    colormap_name: str = "viridis",
    rescale: tuple[float, float] | None = None,
    scale: int = 1,
    algorithm: BaseAlgorithm | None = None,
    **kwargs: str | int,
) -> str:
    """Adds a DataArray to the TiTiler server and returns a URL template.

    The TiTiler server is lazily started when the first DataArray is added.

    Args:
        data_array: An Xarray DataArray to dynamically tile for visualization.
        colormap_name: A rio-tiler colormap name.
        rescale: Comma (',') delimited Min,Max range.
        scale: Tile size scale. Default 1 corresponds to 256*256px tiles.
        algorithm: Custom TiTiler algorithm name, e.g. "hillshade".
        kwargs: Additional query parameters to include in the TiTiler request URL.

    Returns:
        A URL pointing to the new tile endpoint.
    """
    return await _get_server().add_data_array(
        data_array,
        colormap_name=colormap_name,
        rescale=rescale,
        scale=scale,
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
