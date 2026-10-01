import json

import pytest
from rio_tiler.types import ColorMapType

from jupyter_tiler.titiler._query import _build_tile_query_params


class TestBuildTileQueryParams:
    def _build(  # noqa: PLR0913
        self,
        *,
        colormap_name: str | None = None,
        colormap: ColorMapType | None = None,
        colormap_range: tuple[float, float] | None = None,
        tile_dim_scale: int = 1,
        has_algorithm: bool = False,
        extra_params: dict[str, str | int] | None = None,
    ) -> dict[str, str | int]:
        return _build_tile_query_params(
            colormap_name=colormap_name,
            colormap=colormap,
            colormap_range=colormap_range,
            tile_dim_scale=tile_dim_scale,
            has_algorithm=has_algorithm,
            extra_params=extra_params or {},
        )

    def test_defaults_to_viridis_colormap_name(self) -> None:
        params = self._build()
        assert params["colormap_name"] == "viridis"
        assert "colormap" not in params

    def test_colormap_name_is_used(self) -> None:
        assert self._build(colormap_name="cool")["colormap_name"] == "cool"

    @pytest.mark.parametrize(
        "colormap",
        [
            pytest.param(
                {0: (0, 0, 0, 0), 1: (255, 255, 255, 255)},
                id="discrete-dict",
            ),
            pytest.param(
                [((0.0, 0.5), (255, 0, 0, 255)), ((0.5, 1.0), (0, 0, 255, 255))],
                id="intervals",
            ),
        ],
    )
    def test_custom_colormap_is_json_encoded_and_omits_name(
        self,
        colormap: ColorMapType,
    ) -> None:
        params = self._build(colormap=colormap)
        encoded_colormap = params["colormap"]

        assert "colormap_name" not in params
        assert isinstance(encoded_colormap, str)
        assert json.loads(encoded_colormap) == json.loads(json.dumps(colormap))

    def test_colormap_and_colormap_name_are_mutually_exclusive(self) -> None:
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            self._build(colormap={0: (0, 0, 0, 0)}, colormap_name="viridis")

    def test_colormap_range_becomes_rescale(self) -> None:
        assert self._build(colormap_range=(0, 1))["rescale"] == "0,1"
