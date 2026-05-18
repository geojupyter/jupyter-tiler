import anyio
import pytest
from xarray import DataArray

from jupyter_xarray_tiler.xpublish._server import XpublishServer

from .helpers import check_tile
from .params import params_for_backend


class TestXpublishServer:
    @pytest.mark.asyncio
    async def test_server_is_not_singleton(self) -> None:
        """Test that XpublishServer is not a singleton.

        Previously, we used a singleton pattern for Xpublish server, but not anymore.
        Now, tests depend on being able to create a fresh instance and the end-user is
        protected from starting multiple instances in the public API.
        """
        assert id(XpublishServer()) != id(XpublishServer())
        assert XpublishServer() is not XpublishServer()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("z", "y", "x", "mock_data_array"),
        params_for_backend("xpublish"),
        indirect=["mock_data_array"],
    )
    async def test_add_data_array_works(
        self,
        z: int,
        y: int,
        x: int,
        clean_xpublish_server: XpublishServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that tiles can be accessed after a data array is added to the server."""
        proxy_url = await clean_xpublish_server.add_data_array(
            data_array=mock_data_array,
            colormap_range=(0, 1),
        )

        await check_tile(proxy_url=proxy_url.format(z=z, y=y, x=x))

    @pytest.mark.asyncio
    async def test_add_data_array_returns_valid_tile_url(
        self,
        clean_xpublish_server: XpublishServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that adding a DataArray returns a properly formatted tile URL."""
        tile_url = await clean_xpublish_server.add_data_array(
            data_array=mock_data_array,
        )

        assert tile_url is not None
        assert "/proxy/" in tile_url
        assert f"/{clean_xpublish_server._port}/" in tile_url
        assert "/tiles/WebMercatorQuad/{z}/{y}/{x}" in tile_url


class TestXpublishServerRestart:
    @pytest.mark.asyncio
    async def test_server_started_event_is_cleared_after_stop(
        self,
        clean_xpublish_server: XpublishServer,
    ) -> None:
        """Test that _started is cleared so the server can be restarted."""
        assert clean_xpublish_server._started.is_set()

        with anyio.fail_after(5):
            await clean_xpublish_server.stop()

        assert not clean_xpublish_server._started.is_set()
        assert clean_xpublish_server._port is None
        assert clean_xpublish_server._app is None

    @pytest.mark.asyncio
    async def test_server_binds_to_new_port_after_restart(
        self,
        clean_xpublish_server: XpublishServer,
    ) -> None:
        """Test restarted server binds to a fresh port."""
        port_before_restart = clean_xpublish_server._port

        with anyio.fail_after(5):
            await clean_xpublish_server.stop()
            await clean_xpublish_server.start()

        assert clean_xpublish_server._started.is_set()
        assert clean_xpublish_server._port != port_before_restart

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("z", "y", "x", "mock_data_array"),
        params_for_backend("xpublish"),
        indirect=["mock_data_array"],
    )
    async def test_add_data_array_serves_tiles_after_restart(
        self,
        z: int,
        y: int,
        x: int,
        clean_xpublish_server: XpublishServer,
        mock_data_array: DataArray,
    ) -> None:
        """Test that tiles are accessible from a layer added after a restart."""
        with anyio.fail_after(5):
            await clean_xpublish_server.stop()

        proxy_url = await clean_xpublish_server.add_data_array(
            data_array=mock_data_array,
            colormap_range=(0, 1),
        )

        await check_tile(proxy_url=proxy_url.format(z=z, y=y, x=x))

    @pytest.mark.asyncio
    async def test_stop_tile_server_does_not_hang_during_startup(self) -> None:
        """Test stop() doesn't block if called during startup."""
        server = XpublishServer()

        with anyio.fail_after(5):
            await server.start()
            await server.stop()
