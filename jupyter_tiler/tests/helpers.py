from io import BytesIO

import httpx
import numpy as np
from PIL import Image

from .exceptions import TileIsTransparentError, TileRequestFailedError


async def get_tile_image(*, proxy_url: str) -> Image.Image:
    """Fetch a tile and return it as an RGBA image."""
    url = _proxy_url_to_localhost_url(proxy_url)

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)

    if resp.status_code != 200:  # noqa: PLR2004
        raise TileRequestFailedError(
            status=resp.status_code,
            text=resp.text,
        )

    return Image.open(BytesIO(resp.content)).convert("RGBA")


async def check_tile(*, proxy_url: str, transparent_ok: bool = False) -> None:
    """Check that we can fetch a tile and it looks OK.

    We check that it isn't fully transparent/empty, unless `transparent_ok` in which
    case we just check that we can fetch it.
    """
    img = await get_tile_image(proxy_url=proxy_url)

    if transparent_ok:
        return

    alpha = np.array(img)[:, :, 3]

    is_transparent = alpha.max() == 0
    if is_transparent:
        raise TileIsTransparentError


def _proxy_url_to_localhost_url(proxy_url: str) -> str:
    return f"http://localhost:{proxy_url.removeprefix('/proxy/')}"
