import anyio
import pytest
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer

from .helpers import check_tile
from .params import params_for_backend


class TestTiTilerServer:
    @pytest.mark.asyncio
    async def test_server_is_not_singleton(self) -> None:
        """Test that TiTilerServer is not a singleton.

        Previously, we used a singleton pattern for TiTiler server, but not anymore.
        Now, tests depend on being able to create a fresh instance and the end-user is
        protected from starting multiple instances in the public API.
        """
        assert id(TiTilerServer()) != id(TiTilerServer())
        assert TiTilerServer() is not TiTilerServer()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("z", "y", "x", "mock_data_array"),
        params_for_backend("titiler"),
        indirect=["mock_data_array"],
    )
    async def test_add_data_array_works(
        self,
        z: int,
        y: int,
        x: int,
        clean_titiler_server: TiTilerServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that FastAPI routes are created when a data array is added to server."""
        assert len(clean_titiler_server.routes) == 0

        proxy_url = await clean_titiler_server.add_data_array(
            data_array=mock_data_array
        )

        assert len(clean_titiler_server.routes) > 0

        await check_tile(proxy_url=proxy_url.format(z=z, y=y, x=x))

    @pytest.mark.asyncio
    async def test_add_data_array_returns_valid_tile_url(
        self,
        clean_titiler_server: TiTilerServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that adding a DataArray returns a properly formatted tile URL."""
        tile_url = await clean_titiler_server.add_data_array(data_array=mock_data_array)

        assert tile_url is not None
        assert "/proxy/" in tile_url
        assert f"/{clean_titiler_server._port}/" in tile_url
        assert "/tiles/WebMercatorQuad/{z}/{x}/{y}.png" in tile_url
        assert "colormap_name=viridis" in tile_url
        assert "scale=1" in tile_url


class TestTiTilerServerRestart:
    @pytest.mark.asyncio
    async def test_server_started_event_is_cleared_after_stop(
        self,
        clean_titiler_server: TiTilerServer,
    ) -> None:
        """Test that _started is cleared so the server can be restarted."""
        assert clean_titiler_server._started.is_set()

        with anyio.fail_after(5):
            await clean_titiler_server.stop()

        assert not clean_titiler_server._started.is_set()
        assert clean_titiler_server._port is None
        assert clean_titiler_server._app is None

    @pytest.mark.asyncio
    async def test_server_binds_to_new_port_after_restart(
        self,
        clean_titiler_server: TiTilerServer,
    ) -> None:
        """Test restarted server binds to a fresh port."""
        port_before_restart = clean_titiler_server._port

        with anyio.fail_after(5):
            await clean_titiler_server.stop()
            await clean_titiler_server.start()

        assert clean_titiler_server._started.is_set()
        assert clean_titiler_server._port != port_before_restart

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("z", "y", "x", "mock_data_array"),
        params_for_backend("titiler"),
        indirect=["mock_data_array"],
    )
    async def test_add_data_array_serves_tiles_after_restart(
        self,
        z: int,
        y: int,
        x: int,
        clean_titiler_server: TiTilerServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that tiles are accessible from a layer added after a restart."""
        with anyio.fail_after(5):
            await clean_titiler_server.stop()

        proxy_url = await clean_titiler_server.add_data_array(
            data_array=mock_data_array
        )

        await check_tile(proxy_url=proxy_url.format(z=z, y=y, x=x))

    @pytest.mark.asyncio
    async def test_stop_tile_server_does_not_hang_during_startup(self) -> None:
        """Test stop() doesn't block if called during startup."""
        server = TiTilerServer()

        with anyio.fail_after(5):
            await server.start()
            await server.stop()
