import json

import anyio
import pytest
from rio_tiler.types import ColorMapType
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer, _build_tile_query_params

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


class TestBuildTileQueryParams:
    def _build(  # noqa: PLR0913
        self,
        *,
        colormap_name: str | None = None,
        colormap: ColorMapType | None = None,
        colormap_range: tuple[float, float] | None = None,
        tile_dim_scale: int = 1,
        has_algorithm: bool = False,
        extra_params: dict[str, str | int] | None = None,
    ) -> dict[str, str | int]:
        return _build_tile_query_params(
            colormap_name=colormap_name,
            colormap=colormap,
            colormap_range=colormap_range,
            tile_dim_scale=tile_dim_scale,
            has_algorithm=has_algorithm,
            extra_params=extra_params or {},
        )

    def test_defaults_to_viridis_colormap_name(self) -> None:
        params = self._build()
        assert params["colormap_name"] == "viridis"
        assert "colormap" not in params

    def test_colormap_name_is_used(self) -> None:
        assert self._build(colormap_name="cool")["colormap_name"] == "cool"

    @pytest.mark.parametrize(
        "colormap",
        [
            pytest.param(
                {0: (0, 0, 0, 0), 1: (255, 255, 255, 255)},
                id="discrete-dict",
            ),
            pytest.param(
                [((0.0, 0.5), (255, 0, 0, 255)), ((0.5, 1.0), (0, 0, 255, 255))],
                id="intervals",
            ),
        ],
    )
    def test_custom_colormap_is_json_encoded_and_omits_name(
        self,
        colormap: ColorMapType,
    ) -> None:
        params = self._build(colormap=colormap)
        assert "colormap_name" not in params
        assert json.loads(params["colormap"]) == json.loads(json.dumps(colormap))

    def test_colormap_and_colormap_name_are_mutually_exclusive(self) -> None:
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            self._build(colormap={0: (0, 0, 0, 0)}, colormap_name="viridis")

    def test_colormap_range_becomes_rescale(self) -> None:
        assert self._build(colormap_range=(0, 1))["rescale"] == "0,1"
