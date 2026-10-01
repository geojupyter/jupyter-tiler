from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rio_tiler.types import ColorMapType


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
