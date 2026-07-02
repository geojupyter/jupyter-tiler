import uuid
from collections.abc import Callable
from urllib.parse import urlencode

from fastapi import FastAPI
from rio_tiler.io.xarray import XarrayReader
from rio_tiler.models import ImageData
from titiler.core.algorithm import algorithms as default_algorithms
from titiler.core.algorithm.base import BaseAlgorithm
from titiler.core.dependencies import DefaultDependency
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.core.factory import TilerFactory
from xarray import DataArray

from jupyter_tiler._base_server import _FastApiTileServer
from jupyter_tiler.constants._messages import (
    _found_bug_message,
    _not_initialized_message,
)
from jupyter_tiler.titiler._xarray_stac_backend import XarraySTACTilerFactory


class TiTilerServer(_FastApiTileServer):
    """Manage a TiTiler FastAPI server instance.

    In practice, there should only ever be a single instance of this class.
    But this class is not a singleton: the public API handles this under the hood via a
    private function which holds a single instance in its cache.
    """

    def _init_fastapi_app(self) -> FastAPI:
        app = FastAPI(
            openapi_url="/",
            docs_url=None,
            redoc_url=None,
        )
        add_exception_handlers(app, DEFAULT_STATUS_CODES)
        return app

    async def add_data_array(
        self,
        data_array: DataArray,
        *,
        colormap_name: str = "viridis",
        colormap_range: tuple[float, float] | None = None,
        tile_dim_scale: int = 1,
        algorithm: BaseAlgorithm | None = None,
        **kwargs: str | int,
    ) -> str:
        """Add a data array to the TiTiler server."""
        await self.start()

        if self._port is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        source_id = str(uuid.uuid4())
        self._add_data_array_route(
            source_id=source_id,
            data_array=data_array,
            algorithm=algorithm,
        )

        _params = {
            "scale": str(tile_dim_scale),
            "colormap_name": colormap_name,
            "reproject": "max",
            **kwargs,
        }
        if colormap_range is not None:
            _params["rescale"] = f"{colormap_range[0]},{colormap_range[1]}"
        if algorithm is not None:
            _params["algorithm"] = "algorithm"

        return (
            f"{self._base_url}/{source_id}/tiles/WebMercatorQuad"
            "/{z}/{x}/{y}.png"
            f"?{urlencode(_params)}"
        )

    async def add_stac_array(
        self,
        stac_url: str,
        array_to_image: Callable[[DataArray], ImageData],
        collection_id: str,
        *,
        assets: list[str] | None = None,
        max_items: int = 4,
        resolution_scale: float = 2.0,
        resampling: str = "nearest",
        viewport_width: int = 640,
        viewport_height: int = 360,
        viewport_resampling: str = "linear",
        **kwargs: str | int,
    ) -> str:
        """Add a data array to the TiTiler server."""
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        if resolution_scale <= 0:
            raise ValueError("resolution_scale must be > 0")
        if viewport_width < 0 or viewport_height < 0:
            raise ValueError("viewport_width and viewport_height must be >= 0")

        await self.start()

        if self._port is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        source_id = str(uuid.uuid4())
        self._add_stac_array_route(
            source_id=source_id,
            stac_url=stac_url,
            assets=assets,
            max_items=max_items,
            resolution_scale=resolution_scale,
            resampling=resampling,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            viewport_resampling=viewport_resampling,
            array_to_image=array_to_image,
        )

        query = urlencode(kwargs)
        query_suffix = f"?{query}" if query else ""

        return (
            f"{self._base_url}/{source_id}/collections/{collection_id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
            f"{query_suffix}"
        )

    def _add_data_array_route(  # type: ignore[override]
        self,
        *,
        source_id: str,
        data_array: DataArray,
        algorithm: BaseAlgorithm | None = None,
    ) -> None:
        if self._app is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        algorithms = default_algorithms
        if algorithm is not None:
            algorithms = default_algorithms.register({"algorithm": algorithm})

        tiler = TilerFactory(
            router_prefix=f"/{source_id}",
            reader=XarrayReader,
            path_dependency=lambda: data_array,
            reader_dependency=DefaultDependency,
            process_dependency=algorithms.dependency,
        )
        self._app.include_router(tiler.router, prefix=f"/{source_id}")

    def _add_stac_array_route(  # type: ignore[override]
        self,
        *,
        source_id: str,
        stac_url: str,
        array_to_image: Callable[[DataArray], ImageData],
        assets: list[str] | None = None,
        max_items: int = 4,
        resolution_scale: float = 2.0,
        resampling: str = "nearest",
        viewport_width: int = 0,
        viewport_height: int = 0,
        viewport_resampling: str = "linear",
    ) -> None:
        if self._app is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        # titiler.stacapi dependencies read the STAC API URL from app.state.stac_url.
        self._app.state.stac_url = stac_url

        tiler = XarraySTACTilerFactory(
            stac_url=stac_url,
            assets=assets,
            max_items=max_items,
            resolution_scale=resolution_scale,
            resampling=resampling,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            viewport_resampling=viewport_resampling,
            array_to_image=array_to_image,
            router_prefix=f"/{source_id}",
        )
        self._app.include_router(tiler.router, prefix=f"/{source_id}")
