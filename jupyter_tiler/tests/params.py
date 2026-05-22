"""Shared test parameters for tile tests."""

import itertools
from collections.abc import Generator
from typing import Literal

import pytest
from _pytest.mark.structures import ParameterSet

from .exceptions import (
    BaseTestError,
    TileIsTransparentError,
    TileRequestFailedError,
)

type Backend = Literal["xpublish", "titiler"]

TILES = {  # TODO: Make these Enums for better typing
    "farzoom": (1, 1, 1),
    "midzoom": (4, 9, 4),
    "closezoom": (8, 69, 169),
}

COORD_NAME_SCHEMES = {  # TODO: Make these Enums for better typing
    "latitude/longitude": ("latitude", "longitude"),
    "lat/lon": ("lat", "lon"),
    "y/x": ("y", "x"),
}

# Mapping of coordinate tile and coordinate scheme to expected result (Exception / pass)
type Expectations = dict[tuple[str, str], type[BaseTestError] | None]

# IMPORTANT: Xpublish and TiTiler both work with `latitude`/`longitude` coordinate
# names. Otherwise, their behavior is inconsistent.
#
# See:
#
# * <https://github.com/earth-mover/xpublish-tiles/issues/206>
# * <https://github.com/developmentseed/titiler/issues/1339>
EXPECTATIONS: dict[Backend, Expectations] = {
    "xpublish": {
        ("farzoom", "latitude/longitude"): None,
        ("farzoom", "lat/lon"): None,
        ("farzoom", "y/x"): TileRequestFailedError,
        ("midzoom", "latitude/longitude"): None,
        ("midzoom", "lat/lon"): None,
        ("midzoom", "y/x"): TileRequestFailedError,
        ("closezoom", "latitude/longitude"): TileIsTransparentError,
        ("closezoom", "lat/lon"): TileIsTransparentError,
        ("closezoom", "y/x"): TileRequestFailedError,
    },
    "titiler": {
        ("farzoom", "latitude/longitude"): None,
        ("farzoom", "lat/lon"): TileRequestFailedError,
        ("farzoom", "y/x"): None,
        ("midzoom", "latitude/longitude"): None,
        ("midzoom", "lat/lon"): TileRequestFailedError,
        ("midzoom", "y/x"): None,
        ("closezoom", "latitude/longitude"): None,
        ("closezoom", "lat/lon"): TileRequestFailedError,
        ("closezoom", "y/x"): None,
    },
}

for backend, expectations in EXPECTATIONS.items():
    keys = set(expectations.keys())
    all_combos = set(itertools.product(TILES.keys(), COORD_NAME_SCHEMES.keys()))
    missing_combos = all_combos - keys
    if missing_combos:
        raise RuntimeError(
            f"All combinations not expressed for backend {backend} in EXPECTATIONS."
            f" Missing: {missing_combos}"
        )


def params_for_backend(backend: Backend) -> Generator[ParameterSet, None, None]:
    """Generate all test parameters for a backend from EXPECTATIONS.

    Yields pytest.param objects with appropriate marks.

    E.g.:

        pytest.param(
            1, 1, 1, "y/x",
            marks=pytest.mark.xfail(
                reason="Some reason",
                raises=SomeException,
            )
        ))
    """
    expectations = EXPECTATIONS[backend]

    for (tile_id, coord_name_scheme_id), expected_exception in expectations.items():
        z, y, x = TILES[tile_id]
        coord_dim_names = COORD_NAME_SCHEMES[coord_name_scheme_id]

        test_id = f"{tile_id}-{coord_name_scheme_id}"

        if expected_exception is None:
            yield pytest.param(z, y, x, coord_dim_names, id=test_id)
        else:
            yield pytest.param(
                z,
                y,
                x,
                coord_dim_names,
                marks=pytest.mark.xfail(
                    reason=expected_exception.reason,
                    raises=expected_exception,
                ),
                id=test_id,
            )
