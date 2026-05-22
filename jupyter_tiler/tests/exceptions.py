class BaseTestError(Exception):
    reason: str

    def __init__(self, *, detail: str | None = None) -> None:
        message = self.reason
        if detail:
            message = f"{message}\n{detail}"

        super().__init__(message)


class TileIsTransparentError(BaseTestError):
    """A tile served during testing is fully transparent."""

    reason = (
        "Transparent tile received."
        " See <https://github.com/earth-mover/xpublish-tiles/issues/206#issuecomment-4015544811>"
    )


class TileRequestFailedError(BaseTestError):
    """A request for a tile during testing returned a non-200 response code."""

    reason = (
        "Non-200 HTTP response code received."
        " See <https://github.com/earth-mover/xpublish-tiles/issues/206#issuecomment-4015544811>"
    )

    def __init__(self, *, status: int, text: str) -> None:
        super().__init__(
            detail=f"Status: {status}\nBody: {text}",
        )
