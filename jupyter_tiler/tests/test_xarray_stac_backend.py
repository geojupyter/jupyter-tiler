import asyncio
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import morecantile
import numpy as np
import pystac
import pytest
import rasterio
import xarray as xr
from PIL import Image
from rasterio.transform import from_origin
from rio_tiler.models import ImageData
from titiler.core.resources.enums import ImageType
from titiler.core.utils import render_image
from xarray import DataArray

from jupyter_tiler.titiler import _xarray_stac_backend as backend_module
from jupyter_tiler.titiler._server import TiTilerServer
from jupyter_tiler.titiler._xarray_stac_backend import (
    _apply_nodata_transparency,
    _filter_items_by_nodata,
    _nodata_alpha_mask,
    _sign_planetary_computer_items,
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


# Extent of the fake STAC items: lon [-3.0, -2.5], lat [47.2, 47.7]
_FAKE_ITEM_BOUNDS = (-3.0, 47.2, -2.5, 47.7)


def _make_local_stac_items(tmp_path: Path, n_items: int = 3) -> list[pystac.Item]:
    """Create local COGs + matching pystac items, so tile requests need no network."""
    west, south, east, north = _FAKE_ITEM_BOUNDS
    res = (east - west) / 200
    transform = from_origin(west, north, res, res)
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }
    items = []
    for i in range(n_items):
        tif = tmp_path / f"item-{i}.tif"
        data = (np.random.default_rng(i).random((200, 200)) * 1000).astype("uint16")
        data[80:120, 80:120] = 0  # nodata patch overlapping the center tile
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            width=200,
            height=200,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
            nodata=0,
        ) as ds:
            ds.write(data, 1)
        item = pystac.Item(
            id=f"item-{i}",
            collection="test",
            geometry=geometry,
            bbox=[west, south, east, north],
            datetime=datetime(2024, 6, 15, tzinfo=UTC),
            properties={},
        )
        proj_fields = {
            "proj:epsg": 4326,
            "proj:shape": [200, 200],
            "proj:transform": [res, 0, west, 0, -res, north],
            "raster:bands": [{"nodata": 0}],
        }
        for band in ("green", "nir"):
            item.add_asset(band, pystac.Asset(href=str(tif), extra_fields=proj_fields))
        items.append(item)
    return items


def _ndwi(array: DataArray) -> ImageData:
    if "time" in array.dims:
        array = array.isel(time=0)
    green = array.sel(band="green").data.astype(np.float32)
    nir = array.sel(band="nir").data.astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndwi = np.nan_to_num(np.asarray(ndwi), nan=-1.0, posinf=1.0, neginf=-1.0)
    pixels = (np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0) * 255).astype(np.uint8)
    return ImageData(pixels[np.newaxis, :, :])


@pytest.mark.asyncio
async def test_multiple_stac_tiles_load_concurrently(
    clean_titiler_server: TiTilerServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch a burst of STAC tiles at once (like a map viewport) and check they
    all load: 200 responses, valid PNGs, and real (non-transparent) pixels."""
    items = _make_local_stac_items(tmp_path)

    # No STAC API in tests: fake the ItemSearch the backend performs, so the
    # real get_assets (including Planetary Computer SAS signing) still runs.
    search_calls: list[dict[str, Any]] = []
    sign_calls: list[list[pystac.Item]] = []

    class FakeItemSearch:
        def __init__(
            self,
            _url: str,
            _stac_io: Any = None,  # noqa: ANN401
            **params: Any,  # noqa: ANN401
        ) -> None:
            search_calls.append(params)

        def items(self) -> list[pystac.Item]:
            return items

    monkeypatch.setattr(backend_module, "ItemSearch", FakeItemSearch)

    # Planetary Computer URLs trigger SAS signing; spy on the signer and make
    # the token fetch a no-op (local COG hrefs are not blob URLs anyway)
    real_sign = backend_module._sign_planetary_computer_items

    def spy_sign(items: list[pystac.Item]) -> list[pystac.Item]:
        sign_calls.append(items)
        return real_sign(items)

    monkeypatch.setattr(backend_module, "_sign_planetary_computer_items", spy_sign)
    monkeypatch.setattr(backend_module, "_pc_sas_token", lambda *_args: "sig=fake")

    tile_url = await clean_titiler_server.add_stac_array(
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        array_to_image=_ndwi,
        collection_id="test",
        assets=["green", "nir"],
        max_items=3,
        max_nodata_ratio=0.9,
    )
    base_url = f"http://localhost:{tile_url.removeprefix('/proxy/')}"

    tms = morecantile.tms.get("WebMercatorQuad")
    center = tms.tile(-2.75, 47.45, 12)
    tiles = [
        (12, center.x + dx, center.y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    ]

    async def fetch(
        client: httpx.AsyncClient, z: int, x: int, y: int
    ) -> httpx.Response:
        url = base_url.format(z=z, x=x, y=y)
        return await client.get(url, timeout=60)

    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*(fetch(client, *t) for t in tiles))
        # a tile far outside the item extent must fail fast with a clean
        # client error, not hang or 500
        outside = await fetch(client, 12, center.x + 10, center.y)

    assert outside.status_code == 404

    # the 3x3 burst was coalesced into a single STAC search over the union
    # bounds; the out-of-extent tile fired afterwards forms its own batch
    assert len(search_calls) == 2
    # the Planetary Computer URL triggered SAS signing of the returned items
    assert sign_calls

    assert len(responses) == 9
    any_transparent = False
    for resp in responses:
        assert resp.status_code == 200, resp.text[:200]
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        with Image.open(io.BytesIO(resp.content)) as img:
            rgba = np.asarray(img.convert("RGBA"))
        assert rgba.shape == (256, 256, 4)
        # the fake items cover these tiles, so pixels must not be all transparent
        assert rgba[:, :, 3].max() == 255
        # the nodata corner of the rasters must be rendered transparent somewhere
        any_transparent |= bool((rgba[:, :, 3] == 0).any())
    assert any_transparent


def test_sign_planetary_computer_items_signs_blob_hrefs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_module, "_pc_sas_token", lambda *_args: "sig=fake")
    item = pystac.Item(
        id="item-0",
        collection="test",
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[0, 0, 1, 1],
        datetime=datetime(2024, 6, 15, tzinfo=UTC),
        properties={},
    )
    blob_href = "https://sentinel2l2a01.blob.core.windows.net/sentinel-s2-l2/x/y.tif"
    item.add_asset("B03", pystac.Asset(href=blob_href))
    item.add_asset("preview", pystac.Asset(href="https://example.com/preview.png"))
    item.add_asset("signed", pystac.Asset(href=blob_href + "?sig=already"))

    out = _sign_planetary_computer_items([item])

    assert out[0].assets["B03"].href == blob_href + "?sig=fake"
    assert out[0].assets["preview"].href == "https://example.com/preview.png"
    assert out[0].assets["signed"].href == blob_href + "?sig=already"
