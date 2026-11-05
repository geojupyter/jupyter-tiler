import io

import numpy as np
import pytest
import xarray as xr
from PIL import Image
from rio_tiler.models import ImageData
from titiler.core.resources.enums import ImageType
from titiler.core.utils import render_image
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer
from jupyter_tiler.titiler._xarray_stac_backend import (
    _apply_nodata_transparency,
    _filter_items_by_nodata,
    _nodata_alpha_mask,
)


def _stack_with_nodata(nodata_fractions: list[float]) -> xr.DataArray:
    """Build a synthetic stackstac-like array, one time slice per item."""
    slices = []
    for fraction in nodata_fractions:
        data = np.ones((4, 4), dtype="float64")
        n_nodata = round(fraction * data.size)
        data.flat[:n_nodata] = np.nan
        slices.append(data)
    return xr.DataArray(
        np.stack(slices),
        dims=["time", "y", "x"],
        coords={"time": [f"item-{i}" for i in range(len(slices))]},
    )


def test_filter_items_by_nodata_drops_mostly_nodata_items() -> None:
    stack = _stack_with_nodata([0.0, 0.5, 1.0])

    filtered, keep = _filter_items_by_nodata(stack, max_nodata_ratio=0.9)

    assert keep.tolist() == [True, True, False]
    assert filtered.sizes["time"] == 2
    assert filtered["time"].values.tolist() == ["item-0", "item-1"]


def test_filter_items_by_nodata_keeps_everything_below_threshold() -> None:
    stack = _stack_with_nodata([0.0, 0.25])

    filtered, keep = _filter_items_by_nodata(stack, max_nodata_ratio=0.5)

    assert keep.tolist() == [True, True]
    assert filtered.sizes["time"] == 2


def test_filter_items_by_nodata_without_time_dim_is_noop() -> None:
    array = xr.DataArray(np.ones((4, 4)), dims=["y", "x"])

    filtered, keep = _filter_items_by_nodata(array, max_nodata_ratio=0.1)

    assert keep.tolist() == [True]
    assert filtered is array


def test_filter_items_by_nodata_respects_band_dim() -> None:
    # item-0 is nodata in one of two bands everywhere -> 50% nodata overall
    band0 = np.ones((2, 2))
    band1 = np.full((2, 2), np.nan)
    stack = xr.DataArray(
        np.stack([np.stack([band0, band1])]),
        dims=["time", "band", "y", "x"],
    )

    _, keep = _filter_items_by_nodata(stack, max_nodata_ratio=0.6)
    assert keep.tolist() == [True]

    _, keep = _filter_items_by_nodata(stack, max_nodata_ratio=0.4)
    assert keep.tolist() == [False]


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.5])
@pytest.mark.asyncio
async def test_add_stac_array_rejects_invalid_max_nodata_ratio(
    clean_titiler_server: TiTilerServer,
    bad_ratio: float,
) -> None:
    def passthrough(_array: DataArray) -> ImageData:
        raise NotImplementedError

    with pytest.raises(ValueError, match="max_nodata_ratio"):
        await clean_titiler_server.add_stac_array(
            stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
            array_to_image=passthrough,
            collection_id="sentinel-2-l2a",
            max_nodata_ratio=bad_ratio,
        )


def _stack_with_nodata_pixels() -> xr.DataArray:
    """Two items x two bands; item-1 is fully nodata."""
    data = np.ones((2, 2, 4, 4), dtype="float64")
    data[1, :, :, :] = np.nan
    return xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"time": ["item-0", "item-1"]},
    )


def test_nodata_alpha_mask_marks_all_nodata_pixels() -> None:
    stack = _stack_with_nodata_pixels()  # item-1 is fully nodata
    stack[0, 0, 0, 0] = np.nan  # one band of item-0 nodata at (0, 0)
    stack[0, :, 2, 3] = np.nan  # all bands of item-0 nodata at (2, 3)

    alpha = _nodata_alpha_mask(stack)

    # a pixel is nodata only if every time slice and band is null there
    expected = np.full((4, 4), 255, dtype="uint8")
    expected[2, 3] = 0
    np.testing.assert_array_equal(alpha, expected)


def test_apply_nodata_transparency_sets_alpha_mask() -> None:
    stack = _stack_with_nodata_pixels()
    image = ImageData(np.zeros((1, 4, 4), dtype="uint8"))

    result = _apply_nodata_transparency(image, stack)

    assert result.alpha_mask is not None
    np.testing.assert_array_equal(
        result.alpha_mask, np.full((4, 4), 255, dtype="uint8")
    )


def test_apply_nodata_transparency_combines_with_existing_alpha() -> None:
    stack = _stack_with_nodata_pixels()
    existing = np.full((4, 4), 255, dtype="uint8")
    existing[2, 2] = 0  # array_to_image already masked this pixel
    image = ImageData(np.zeros((1, 4, 4), dtype="uint8"), alpha_mask=existing)

    result = _apply_nodata_transparency(image, stack)

    assert result.alpha_mask is not None
    assert result.alpha_mask[2, 2] == 0
    assert result.alpha_mask[0, 0] == 255


def test_apply_nodata_transparency_skips_on_grid_mismatch() -> None:
    stack = _stack_with_nodata_pixels()
    image = ImageData(np.zeros((1, 8, 8), dtype="uint8"))

    result = _apply_nodata_transparency(image, stack)

    assert result.alpha_mask is None


def test_nodata_pixels_render_transparent_in_png() -> None:
    stack = _stack_with_nodata_pixels()
    stack[0, :, 0, :] = np.nan  # whole first row nodata in every band of item-0
    pixels = np.full((1, 4, 4), 7, dtype="uint8")
    image = _apply_nodata_transparency(ImageData(pixels), stack)

    content, media_type = render_image(image, output_format=ImageType.png)

    assert media_type == "image/png"
    with Image.open(io.BytesIO(content)) as img:
        pixels = np.asarray(img.convert("RGBA"))
    assert pixels[0, 0, 3] == 0  # nodata row is transparent
    assert pixels[2, 0, 3] == 255  # valid pixel is opaque
