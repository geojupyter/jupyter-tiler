import numpy as np
import pytest
import xarray as xr
from rio_tiler.models import ImageData
from xarray import DataArray

from jupyter_tiler.titiler._server import TiTilerServer
from jupyter_tiler.titiler._xarray_stac_backend import _filter_items_by_nodata


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
