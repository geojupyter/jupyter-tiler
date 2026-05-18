from io import BytesIO

import httpx
import numpy as np
from PIL import Image

from .exceptions import TileIsTransparentError, TileRequestFailedError


async def check_tile(*, proxy_url: str, transparent_ok: bool = False) -> None:
    url = _proxy_url_to_localhost_url(proxy_url)

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)

    if resp.status_code != 200:  # noqa: PLR2004
        raise TileRequestFailedError(
            status=resp.status_code,
            text=resp.text,
        )

    if not transparent_ok:
        img = Image.open(BytesIO(resp.content))
        alpha = np.array(img.convert("RGBA"))[:, :, 3]

        is_transparent = alpha.max() == 0
        if is_transparent:
            raise TileIsTransparentError


def _proxy_url_to_localhost_url(proxy_url: str) -> str:
    return f"http://localhost:{proxy_url.removeprefix('/proxy/')}"
