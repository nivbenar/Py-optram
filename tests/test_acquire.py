import json
from pathlib import Path
import pytest
from shapely.geometry import shape
import pyoptram.acquire as acquire
from pyoptram.acquire import load_evalscript
def _polygon_feature(coordinates):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
    }
def _two_feature_aoi(overlap=False):
    second_min_x = 1.0 if overlap else 2.0
    return {
        "type": "FeatureCollection",
        "features": [
            _polygon_feature(
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
            ),
            _polygon_feature(
                [[second_min_x, 0.0], [3.0, 0.0], [3.0, 2.0],
                 [second_min_x, 2.0], [second_min_x, 0.0]]
            ),
        ],
    }
def test_load_aoi_unions_feature_collection_dict():
    geometry = shape(acquire.load_aoi(_two_feature_aoi()))
    assert geometry.bounds == (0.0, 0.0, 3.0, 2.0)
    assert geometry.area == 6.0
def test_acquisition_rejects_equal_dates(tmp_path):
    with pytest.raises(ValueError, match="to_date must be later than from_date"):
        acquire.acquire_optram_inputs(
            aoi=(0, 0, 1, 1),
            from_date="2024-01-01",
            to_date="2024-01-01",
            output_dir=tmp_path,
        )
@pytest.fixture
def credential_file(monkeypatch, tmp_path):
    path = tmp_path / "CDSE" / "cdse_credentials.json"
    monkeypatch.setattr(acquire, "_cdse_credentials_file", lambda: path)
    return path
def test_store_and_retrieve_credentials(credential_file):
    acquire.store_cdse_credentials("client-id", "client-secret")
    assert credential_file.read_text(encoding="utf-8") == (
        '[{"clientid": "client-id", "secret": "client-secret"}]'
    )
    assert acquire.retrieve_cdse_credentials() == {
        "clientid": "client-id",
        "secret": "client-secret",
    }
@pytest.mark.parametrize(
    "veg_index, formula",
    [
        ("NDVI", "(sample.B08 - sample.B04) / (sample.B08 + sample.B04)"),
        (
            "SAVI",
            "1.5 * (sample.B08 - sample.B04) / "
            "(sample.B08 + sample.B04 + 0.5)",
        ),
        (
            "MSAVI",
            "(2 * sample.B08 + 1 - "
            "Math.sqrt(Math.pow(2 * sample.B08 + 1, 2) - "
            "8 * (sample.B08 - sample.B04))) / 2",
        ),
    ],
)
def test_load_evalscript_uses_roptram_vi_formula(veg_index, formula):
    script = load_evalscript(veg_index)
    assert 'bands: ["B04", "B08"]' in script
    assert formula in script
    assert 'sampleType: "FLOAT32"' in script
def test_resolution_output_matches_cdse_centroid_conversion():
    geometry = acquire.load_aoi(
        (12.292349, 47.810849, 12.569037, 47.967123)
    )
    output = acquire._resolution_output(geometry, 10)
    assert output == pytest.approx(
        {
            "resx": 0.00013371609330541562,
            "resy": 0.00008993763143935762,
        }
    )
def test_resolution_output_rejects_more_than_2500_pixels():
    with pytest.raises(ValueError, match="exceeds the allowed maximum"):
        acquire._resolution_output(acquire.load_aoi((0, 0, 0.23, 0.1)), 10)
class _DownloadResponse:
    status_code = 200
    text = ""
    content = b"tiff"
    def raise_for_status(self):
        return None
def _ring(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
def _scene(number, date="2024-01-02", cloud=1, tile="36RXV", top=1.0):
    return {
        "type": "Feature",
        "id": f"S2A_TEST_{number:03d}_{tile}",
        "properties": {
            "datetime": f"{date}T10:20:30Z",
            "eo:cloud_cover": cloud,
        },
        "geometry": {"type": "Polygon", "coordinates": [_ring(0, 0, 1, top)]},
    }
class _CatalogResponse:
    status_code = 200
    text = ""
    def __init__(self, features, next_page=None):
        self.features = features
        self.next_page = next_page
    def raise_for_status(self):
        return None
    def json(self):
        context = {} if self.next_page is None else {"next": self.next_page}
        return {"features": self.features, "context": context}
def _mock_catalog_pages(monkeypatch, pages):
    calls = []
    def post(*args, **kwargs):
        calls.append(dict(kwargs["json"]))
        return pages[len(calls) - 1]
    monkeypatch.setattr(acquire.requests, "post", post)
    monkeypatch.setattr(acquire, "_area_coverage", lambda *args: 100.0)
    return calls
def test_catalog_paginates_beyond_one_hundred_scenes(monkeypatch):
    calls = _mock_catalog_pages(
        monkeypatch,
        [
            _CatalogResponse([_scene(i) for i in range(100)], "page-2"),
            _CatalogResponse([_scene(i) for i in range(100, 125)]),
        ],
    )
    scenes = acquire.search_catalog(
        acquire.load_aoi((0, 0, 1, 1)),
        "2024-01-01",
        "2024-01-03",
        "token",
    )
    assert len(scenes) == 125
    assert "next" not in calls[0]
    assert calls[1]["next"] == "page-2"
def test_catalog_cloud_filter_is_strict(monkeypatch):
    _mock_catalog_pages(
        monkeypatch,
        [_CatalogResponse([_scene(1, cloud=11.999), _scene(2, cloud=12)])],
    )
    scenes = acquire.search_catalog(
        acquire.load_aoi((0, 0, 1, 1)),
        "2024-01-01",
        "2024-01-03",
        "token",
        max_cloud=12,
    )
    assert [scene["id"] for scene in scenes] == ["S2A_TEST_001_36RXV"]
def test_tile_filter_uses_case_sensitive_source_id_substring(monkeypatch):
    _mock_catalog_pages(
        monkeypatch,
        [
            _CatalogResponse(
                [
                    _scene(1, tile="36RXV_EXTRA"),
                    _scene(2, tile="36rxv"),
                    _scene(3, tile="36RXW"),
                ]
            )
        ],
    )
    scenes = acquire.search_catalog(
        acquire.load_aoi((0, 0, 1, 1)),
        "2024-01-01",
        "2024-01-03",
        "token",
        tile="36RXV",
    )
    assert [scene["id"] for scene in scenes] == ["S2A_TEST_001_36RXV_EXTRA"]
def test_s2_area_coverage_rounds_to_three_decimals():
    aoi = acquire._s2_polygon(acquire.load_aoi((0, 0, 1, 1)))
    retained = acquire._area_coverage(
        aoi, {"type": "Polygon", "coordinates": [_ring(0, 0, 1, 0.989995)]}
    )
    rejected = acquire._area_coverage(
        aoi, {"type": "Polygon", "coordinates": [_ring(0, 0, 1, 0.989985)]}
    )
    assert retained == 99.000
    assert rejected == 98.999
def test_catalog_coverage_comparison_is_inclusive(monkeypatch):
    _mock_catalog_pages(
        monkeypatch, [_CatalogResponse([_scene(1), _scene(2)])]
    )
    coverages = iter([99.000, 98.999])
    monkeypatch.setattr(acquire, "_area_coverage", lambda *args: next(coverages))
    scenes = acquire.search_catalog(
        acquire.load_aoi((0, 0, 1, 1)),
        "2024-01-01",
        "2024-01-03",
        "token",
    )
    assert [scene["areaCoverage"] for scene in scenes] == [99.000]
def test_full_period_preserves_catalog_order(monkeypatch):
    _mock_catalog_pages(
        monkeypatch,
        [_CatalogResponse([_scene(1, "2022-06-01"), _scene(2, "2020-06-01")])],
    )
    scenes = acquire.search_catalog(
        acquire.load_aoi((0, 0, 1, 1)),
        "2020-01-01",
        "2022-12-31",
        "token",
        period="full",
    )
    assert [scene["id"] for scene in scenes] == [
        "S2A_TEST_001_36RXV",
        "S2A_TEST_002_36RXV",
    ]
def test_seasonal_filter_same_year_windows_and_boundaries():
    scenes = [
        _scene(1, "2020-04-01"),
        _scene(2, "2020-09-30"),
        _scene(3, "2021-03-31"),
        _scene(4, "2021-04-01"),
        _scene(5, "2021-09-30"),
        _scene(6, "2021-10-01"),
    ]
    selected = acquire._seasonal_filter(scenes, "2020-04-01", "2021-09-30")
    assert [_scene_date["id"] for _scene_date in selected] == [
        "S2A_TEST_005_36RXV",
        "S2A_TEST_004_36RXV",
        "S2A_TEST_002_36RXV",
        "S2A_TEST_001_36RXV",
    ]
def test_download_uses_whole_day_and_most_recent(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(
        acquire.requests,
        "post",
        lambda *args, **kwargs: requests.append(kwargs["json"])
        or _DownloadResponse(),
    )
    acquire.download_index(
        acquire.load_aoi((0, 0, 1, 1)),
        "2024-01-02T10:20:30Z",
        "NDVI",
        tmp_path / "ndvi.tif",
        "token",
        width=100,
        height=100,
    )
    data_filter = requests[0]["input"]["data"][0]["dataFilter"]
    assert data_filter == {
        "timeRange": {
            "from": "2024-01-02T00:00:00.000Z",
            "to": "2024-01-03T00:00:00.000Z",
        },
        "mosaickingOrder": "mostRecent",
    }
@pytest.fixture
def acquisition_mocks(monkeypatch):
    scene = {
        "id": "S2_TEST_T36RXV",
        "properties": {
            "datetime": "2024-01-02T10:20:30Z",
            "eo:cloud_cover": 3,
            "s2:mgrs_tile": "36RXV",
        },
        "areaCoverage": 100.0,
    }
    search_calls = []
    downloads = []
    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: "token")
    monkeypatch.setattr(
        acquire,
        "search_catalog",
        lambda **kwargs: search_calls.append(kwargs) or [scene],
    )
    monkeypatch.setattr(
        acquire,
        "download_index",
        lambda **kwargs: downloads.append(kwargs) or str(kwargs["output_path"]),
    )
    return search_calls, downloads
def test_acquisition_defaults_match_roptram(tmp_path, acquisition_mocks):
    search_calls, downloads = acquisition_mocks
    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="id",
        client_secret="secret",
    )
    assert search_calls[0]["max_cloud"] == 12
    assert search_calls[0]["area_cover"] == 99.0
    assert search_calls[0]["period"] == "full"
    assert search_calls[0]["limit"] is None
    assert [call["script_name"] for call in downloads] == ["NDVI", "STR", "BOA"]
    assert downloads[0]["scl_mask"] is True
    assert all(call["resolution"] == 10 for call in downloads)
    assert all(call["width"] is None and call["height"] is None for call in downloads)
def test_save_image_list_contains_filtered_scenes_before_downloads(
    monkeypatch, tmp_path, acquisition_mocks
):
    _, downloads = acquisition_mocks
    image_list_path = tmp_path / "image_list.json"
    observed = []
    def check_saved(**kwargs):
        observed.append(json.loads(image_list_path.read_text(encoding="utf-8")))
        return str(kwargs["output_path"])
    monkeypatch.setattr(acquire, "download_index", check_saved)
    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="id",
        client_secret="secret",
        save_img_list=True,
    )
    assert not downloads
    assert observed
    assert observed[0]["type"] == "FeatureCollection"
    assert observed[0]["features"][0]["areaCoverage"] == 100.0
