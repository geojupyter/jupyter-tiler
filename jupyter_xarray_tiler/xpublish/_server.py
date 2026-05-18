import uuid
from urllib.parse import urlencode

import xpublish
from fastapi import FastAPI
from xarray import DataArray, Dataset
from xpublish.utils.api import DATASET_ID_ATTR_KEY
from xpublish_tiles.xpublish.tiles.plugin import TilesPlugin

from jupyter_xarray_tiler._base_server import _FastApiTileServer
from jupyter_xarray_tiler.constants._messages import (
    _found_bug_message,
    _not_initialized_message,
)


class XpublishServer(_FastApiTileServer):
    """Manage an xpublish-tiles FastAPI server instance.

    In practice, there should only ever be a single instance of this class.
    But this class is not a singleton: the public API handles this under the hood via a
    private function which holds a single instance in its cache.

    IMPORTANT: An xpublish server's routes are static, as opposed to TiTiler, which
    creates routes every time a data array is added to the server.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rest: xpublish.Rest | None = None

    def _init_fastapi_app(self) -> FastAPI:
        self._rest = xpublish.Rest(
            plugins={"tiles": TilesPlugin()},
        )
        return self._rest.app  # type: ignore[no-any-return]

    async def add_data_array(
        self,
        data_array: DataArray,
        *,
        colormap_range: tuple[float, float] | None = None,
        **kwargs: str | int,
    ) -> str:
        """Add a data array to the Xpublish server."""
        await self.start()

        if self._port is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        # Create route on server for this data array
        source_id = str(uuid.uuid4())
        self._add_data_array_route(
            source_id=source_id,
            data_array=data_array,
        )

        # Construct URL
        _param_defaults = {
            "variables": "data",
            "style": "raster/default",
            "width": 256,
            "height": 256,
            "f": "png",
        }
        _params = {
            **_param_defaults,
            **kwargs,
        }
        if colormap_range is not None:
            _params["colorscalerange"] = f"{colormap_range[0]},{colormap_range[1]}"

        return (
            f"{self._base_url}/datasets/{source_id}/tiles/WebMercatorQuad"
            "/{z}/{y}/{x}"
            f"?{urlencode(_params)}"
        )

    def _add_data_array_route(  # type: ignore[override]
        self,
        *,
        source_id: str,
        data_array: DataArray,
    ) -> None:
        if self._app is None or self._rest is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        dataset: Dataset = data_array.to_dataset(name=data_array.name or "data")
        dataset.attrs[DATASET_ID_ATTR_KEY] = source_id

        # Add dataset to xpublish server
        self._rest._datasets[source_id] = dataset  # noqa: SLF001
