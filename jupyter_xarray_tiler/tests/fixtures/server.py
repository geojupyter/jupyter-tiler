from collections.abc import AsyncGenerator

import pytest_asyncio

from jupyter_xarray_tiler.titiler._server import TiTilerServer
from jupyter_xarray_tiler.xpublish._server import XpublishServer


@pytest_asyncio.fixture
async def clean_titiler_server() -> AsyncGenerator[TiTilerServer]:
    server = TiTilerServer()
    await server.start()
    yield server

    await server.stop()
    del server


@pytest_asyncio.fixture
async def clean_xpublish_server() -> AsyncGenerator[XpublishServer]:
    server = XpublishServer()
    await server.start()
    yield server

    await server.stop()
    del server
