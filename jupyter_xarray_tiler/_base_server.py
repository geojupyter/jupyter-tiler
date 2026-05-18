import asyncio
import contextlib
from abc import ABC, abstractmethod
from asyncio import Event, Lock, Task, create_task
from functools import partial
from typing import Any

import anyio
from anycorn import Config, serve
from anyio import connect_tcp, create_task_group
from fastapi import FastAPI
from fastapi.routing import APIRoute
from xarray import DataArray

from jupyter_xarray_tiler.constants._messages import (
    _found_bug_message,
    _not_initialized_message,
)


class _FastApiTileServer(ABC):
    """Abstract base class for FastAPI tile server implementation.

    Implements serving the FastAPI app asynchronously with anycorn.
    """

    def __init__(self) -> None:
        self._app: FastAPI | None = None
        self._port: int | None = None
        self._task: Task[None] | None = None
        self._started = Event()
        self._lock = Lock()

    @property
    def routes(self) -> list[dict[str, Any]]:
        """Returns a list of available routes on the server."""
        if self._app is None:
            raise RuntimeError(
                _not_initialized_message
                + " If you're seeing this message, you're 'holding it wrong'."
                " Please see the docs!"
            )

        return [
            {"path": route.path, "name": route.name}
            for route in self._app.router.routes
            if isinstance(route, APIRoute)
        ]

    async def start(self) -> None:
        """Start the tile server."""
        async with self._lock:
            if self._started.is_set():
                return

            self._task = create_task(self._start())
            try:
                with anyio.fail_after(30):
                    await self._started.wait()
            except TimeoutError:
                self._task.cancel()
                raise

    @abstractmethod
    async def add_data_array(
        self,
        data_array: DataArray,
        *,
        colormap_range: tuple[float, float] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Add a data array to the tile server.

        Start the tile server if not already started.

        Returns:
            A URL template pointing to the new tile endpoint.
        """
        ...

    async def stop(self) -> None:
        """Stop the tile server."""
        task: Task[None] | None = None
        async with self._lock:
            if self._started.is_set():
                task = self._task

        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _start(self) -> None:
        self._app = self._init_fastapi_app()

        config = Config()
        config.bind = "127.0.0.1:0"

        try:
            async with create_task_group() as tg:
                binds = await tg.start(
                    partial(
                        serve,
                        self._app,  # type: ignore[arg-type]
                        config,
                        mode="asgi",
                    ),
                )

                # Host will always be 127.0.0.1, port is randomized
                host, _port = binds[0][len("http://") :].split(":")
                self._port = int(_port)

                # Poll until the server is accepting connections
                while True:
                    try:
                        await connect_tcp(host, self._port)
                    except OSError:
                        await anyio.sleep(0.05)
                    else:
                        self._started.set()
                        break

        finally:
            # Reset state on exiting task group (i.e. shutdown)
            self._reset_state()

    def _reset_state(self) -> None:
        self._started.clear()
        self._task = None
        self._port = None
        self._app = None

    @abstractmethod
    def _init_fastapi_app(self) -> FastAPI:
        """Initialize a FastAPI object to populate self._app."""
        ...

    @abstractmethod
    def _add_data_array_route(
        self,
        *,
        source_id: str,
        data_array: DataArray,
        **kwargs: Any,  # noqa: ANN401
    ) -> None: ...

    @property
    def _base_url(self) -> str:
        """The URL to the root path of this tiler server instance."""
        if self._port is None:
            raise RuntimeError(f"{_not_initialized_message} {_found_bug_message}")

        return f"/proxy/{self._port}"
