import json
import uuid
from urllib.parse import urlencode

from fastapi import FastAPI
from rio_tiler.io.xarray import XarrayReader
from rio_tiler.types import ColorMapType
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


def _build_tile_query_params(  # noqa: PLR0913
    *,
    colormap_name: str | None,
    colormap: ColorMapType | None,
    colormap_range: tuple[float, float] | None,
    tile_dim_scale: int,
    has_algorithm: bool,
    extra_params: dict[str, str | int],
) -> dict[str, str | int]:
    """Build the query params for a TiTiler tile-endpoint URL."""
    if colormap is not None and colormap_name is not None:
        raise RuntimeError("colormap and colormap_name are mutually exclusive.")

    params: dict[str, str | int] = {
        "scale": str(tile_dim_scale),
        "reproject": "max",
        **extra_params,
    }

    if colormap is not None:
        params["colormap"] = json.dumps(colormap)
    else:
        params["colormap_name"] = colormap_name or "viridis"

    if colormap_range is not None:
        params["rescale"] = f"{colormap_range[0]},{colormap_range[1]}"

    if has_algorithm:
        params["algorithm"] = "algorithm"

    return params


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

    async def add_data_array(  # noqa: PLR0913
        self,
        data_array: DataArray,
        *,
        colormap_name: str | None = None,
        colormap: ColorMapType | None = None,
        colormap_range: tuple[float, float] | None = None,
        tile_dim_scale: int = 1,
        algorithm: BaseAlgorithm | None = None,
        **kwargs: str | int,
    ) -> str:
        """Add a data array to the TiTiler server."""
        params = _build_tile_query_params(
            colormap_name=colormap_name,
            colormap=colormap,
            colormap_range=colormap_range,
            tile_dim_scale=tile_dim_scale,
            has_algorithm=algorithm is not None,
            extra_params=kwargs,
        )

        await self.start()

        if self._port is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        source_id = str(uuid.uuid4())
        self._add_data_array_route(
            source_id=source_id,
            data_array=data_array,
            algorithm=algorithm,
        )

        return (
            f"{self._base_url}/{source_id}/tiles/WebMercatorQuad"
            "/{z}/{x}/{y}.png"
            f"?{urlencode(params)}"
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
