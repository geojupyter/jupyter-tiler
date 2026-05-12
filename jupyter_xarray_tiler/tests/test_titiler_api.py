import pytest
from xarray import DataArray

from jupyter_xarray_tiler.titiler import (
    _get_server,
    add_data_array,
    get_routes,
)

from .helpers import check_tile
from .params import params_for_backend


def test_singleton_ish() -> None:
    """Test that the API only uses one TiTiler server instance."""
    assert id(_get_server()) == id(_get_server())
    assert _get_server() is _get_server()


@pytest.mark.usefixtures("clean_titiler_api")
def test_get_routes_raises_before_server_started() -> None:
    """Test that get_routes raises an error if called before initialization."""
    with pytest.raises(RuntimeError):
        get_routes()


@pytest.mark.usefixtures("clean_titiler_api")
@pytest.mark.parametrize(
    ("z", "y", "x", "mock_data_array"),
    params_for_backend("titiler"),
    indirect=["mock_data_array"],
)
async def test_add_data_array_works(
    z: int,
    y: int,
    x: int,
    mock_data_array: DataArray,
) -> None:
    proxy_url = await add_data_array(data_array=mock_data_array)

    assert len(get_routes()) > 0

    await check_tile(proxy_url=proxy_url.format(z=z, y=y, x=x))
