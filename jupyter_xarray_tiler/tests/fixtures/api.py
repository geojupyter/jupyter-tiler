from collections.abc import AsyncGenerator

import pytest_asyncio

from jupyter_xarray_tiler.titiler import _get_server as _get_titiler_server
from jupyter_xarray_tiler.xpublish import _get_server as _get_xpublish_server


async def _reset_titiler_api_for_testing() -> None:
    # Shutdown the previous server
    server = _get_titiler_server()
    await server.stop_tile_server()
    # Clear the cache so next time we'll get a fresh one
    _get_titiler_server.cache_clear()


@pytest_asyncio.fixture
async def clean_titiler_api() -> AsyncGenerator[None]:
    """Ensure a test's usage of the titiler API is not influenced by other tests.

    I.e., the test will receive a fresh TiTiler server.
    """
    await _reset_titiler_api_for_testing()
    yield
    await _reset_titiler_api_for_testing()


async def _reset_xpublish_api_for_testing() -> None:
    # Shutdown the previous server
    server = _get_xpublish_server()
    await server.stop_tile_server()
    # Clear the cache so next time we'll get a fresh one
    _get_xpublish_server.cache_clear()


@pytest_asyncio.fixture
async def clean_xpublish_api() -> AsyncGenerator[None]:
    """Ensure a test's usage of the xpublish API is not influenced by other tests.

    I.e., the test will receive a fresh xpublish server.
    """
    await _reset_xpublish_api_for_testing()
    yield
    await _reset_xpublish_api_for_testing()
