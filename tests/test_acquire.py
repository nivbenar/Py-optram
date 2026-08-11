### CDSE Acquisition Tests
# Verifies credential handling, evalscript generation, validation, and the
# Sentinel-2 acquisition workflow without live service calls.

import json
from pathlib import Path

import pytest
from shapely.geometry import shape

import pyoptram.acquire as acquire
from pyoptram.acquire import load_evalscript


### Build one polygon feature for AOI union tests.
def _polygon_feature(coordinates):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
    }


### Build a two-feature AOI with touching or overlapping polygons.
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


### AOI and date validation

def test_load_aoi_unions_feature_collection_dict():
    geometry = shape(acquire.load_aoi(_two_feature_aoi()))

    assert geometry.bounds == (0.0, 0.0, 3.0, 2.0)
    assert geometry.area == 6.0


def test_load_aoi_unions_feature_collection_file(tmp_path):
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(_two_feature_aoi()), encoding="utf-8")

    geometry = shape(acquire.load_aoi(path))

    assert geometry.bounds == (0.0, 0.0, 3.0, 2.0)
    assert geometry.area == 6.0


def test_load_aoi_dissolves_overlapping_features():
    geometry = shape(acquire.load_aoi(_two_feature_aoi(overlap=True)))

    assert geometry.geom_type == "Polygon"
    assert geometry.area == 6.0


def test_load_aoi_rejects_empty_feature_collection():
    with pytest.raises(ValueError, match="at least one feature"):
        acquire.load_aoi({"type": "FeatureCollection", "features": []})


def test_load_aoi_preserves_existing_geojson_and_bbox_forms():
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    feature = {"type": "Feature", "properties": {}, "geometry": polygon}

    assert acquire.load_aoi(polygon) is polygon
    assert acquire.load_aoi(feature) is polygon
    assert acquire.load_aoi((0, 0, 1, 1)) == polygon


def test_acquisition_receives_complete_unioned_aoi(monkeypatch, tmp_path):
    search_calls = []
    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: "token")
    monkeypatch.setattr(
        acquire,
        "search_catalog",
        lambda **kwargs: search_calls.append(kwargs) or [],
    )

    acquire.acquire_optram_inputs(
        aoi=_two_feature_aoi(),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="id",
        client_secret="secret",
        save_creds=False,
    )

    geometry = shape(search_calls[0]["aoi_geometry"])
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


def test_acquisition_rejects_reversed_dates(tmp_path):
    with pytest.raises(ValueError, match="to_date must be later than from_date"):
        acquire.acquire_optram_inputs(
            aoi=(0, 0, 1, 1),
            from_date="2024-01-02",
            to_date="2024-01-01",
            output_dir=tmp_path,
        )


### Credential handling

@pytest.fixture
def credential_file(monkeypatch, tmp_path):
    path = tmp_path / "CDSE" / "cdse_credentials.json"
    monkeypatch.setattr(acquire, "_cdse_credentials_file", lambda: path)
    return path


@pytest.mark.parametrize(
    "system, expected",
    [
        ("Linux", ".CDSE/cdse_credentials.json"),
        ("Darwin", "Library/Preferences/.CDSE/cdse_credentials.json"),
    ],
)
def test_credential_file_matches_roptram_home_paths(
    monkeypatch, tmp_path, system, expected
):
    monkeypatch.setattr(acquire.platform, "system", lambda: system)
    monkeypatch.setattr(acquire.Path, "home", lambda: tmp_path)

    assert acquire._cdse_credentials_file() == tmp_path / expected


def test_credential_file_matches_roptram_windows_path(monkeypatch, tmp_path):
    monkeypatch.setattr(acquire.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert acquire._cdse_credentials_file() == (
        tmp_path / "CDSE" / "cdse_credentials.json"
    )


def test_store_and_retrieve_credentials(credential_file):
    acquire.store_cdse_credentials("client-id", "client-secret")

    assert credential_file.read_text(encoding="utf-8") == (
        '[{"clientid": "client-id", "secret": "client-secret"}]'
    )
    assert acquire.retrieve_cdse_credentials() == {
        "clientid": "client-id",
        "secret": "client-secret",
    }


def test_store_credentials_from_roptram_environment_names(
    monkeypatch, credential_file
):
    monkeypatch.setenv("OAUTH_CLIENTID", "environment-id")
    monkeypatch.setenv("OAUTH_SECRET", "environment-secret")

    acquire.store_cdse_credentials()

    assert acquire.retrieve_cdse_credentials() == {
        "clientid": "environment-id",
        "secret": "environment-secret",
    }


def test_store_credentials_from_csv_file(credential_file, tmp_path):
    source = tmp_path / "credentials.csv"
    source.write_text("clientid,secret\nfile-id,file-secret\n", encoding="utf-8")

    assert acquire.store_cdse_credentials_from_file(source) is None
    assert acquire.retrieve_cdse_credentials() == {"clientid": "file-id", "secret": "file-secret"}


def test_acquisition_reuses_file_credentials_after_source_is_removed(
    monkeypatch, tmp_path, credential_file
):
    source = tmp_path / "credentials.csv"
    source.write_text("clientid,secret\nfile-id,file-secret\n", encoding="utf-8")
    acquire.store_cdse_credentials_from_file(source)
    source.unlink()
    token_calls = []
    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: token_calls.append(args) or "token")
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
    )

    assert token_calls == [("file-id", "file-secret")]


@pytest.mark.parametrize(
    "contents, message",
    [
        ("username,password\nuser,password\n", "clientid and secret headers"),
        ("clientid,secret\n\n", "exactly one credential record"),
        ("clientid,secret\nid-one,secret-one\nid-two,secret-two\n", "exactly one credential record"),
        ("clientid,secret\nid-only,\n", "non-empty clientid and secret values"),
    ],
)
def test_store_credentials_from_csv_rejects_invalid_input(
    tmp_path, credential_file, contents, message
):
    source = tmp_path / "credentials.csv"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc_info:
        acquire.store_cdse_credentials_from_file(source)

    assert "secret-one" not in str(exc_info.value)
    assert "secret-two" not in str(exc_info.value)
    assert not credential_file.exists()


def test_store_credentials_from_csv_does_not_print_secret(
    capsys, credential_file, tmp_path
):
    source = tmp_path / "credentials.csv"
    source.write_text("clientid,secret\nfile-id,hidden-secret\n", encoding="utf-8")

    acquire.store_cdse_credentials_from_file(source)

    captured = capsys.readouterr()
    assert "hidden-secret" not in captured.out
    assert "hidden-secret" not in captured.err


def test_retrieve_missing_credentials_warns_without_exposing_secret(
    credential_file,
):
    with pytest.warns(RuntimeWarning, match="Credentials are not available"):
        assert acquire.retrieve_cdse_credentials() is None


def test_acquisition_reuses_stored_credentials(
    monkeypatch, tmp_path, credential_file
):
    acquire.store_cdse_credentials("stored-id", "stored-secret")
    token_calls = []
    monkeypatch.setattr(
        acquire,
        "get_cdse_token",
        lambda client_id, client_secret: token_calls.append(
            (client_id, client_secret)
        )
        or "token",
    )
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
    )

    assert token_calls == [("stored-id", "stored-secret")]


def test_explicit_credentials_override_and_are_saved_after_authentication(
    monkeypatch, tmp_path, credential_file
):
    acquire.store_cdse_credentials("stored-id", "stored-secret")
    observed_file_contents = []

    def fake_token(client_id, client_secret):
        observed_file_contents.append(credential_file.read_text(encoding="utf-8"))
        assert (client_id, client_secret) == ("explicit-id", "explicit-secret")
        return "token"

    monkeypatch.setattr(acquire, "get_cdse_token", fake_token)
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="explicit-id",
        client_secret="explicit-secret",
    )

    assert "stored-secret" in observed_file_contents[0]
    assert acquire.retrieve_cdse_credentials() == {
        "clientid": "explicit-id",
        "secret": "explicit-secret",
    }


def test_failed_authentication_does_not_store_explicit_credentials(
    monkeypatch, tmp_path, credential_file
):
    def fail_token(*args):
        raise requests.HTTPError("authentication failed")

    import requests

    monkeypatch.setattr(acquire, "get_cdse_token", fail_token)

    with pytest.raises(requests.HTTPError, match="authentication failed"):
        acquire.acquire_optram_inputs(
            aoi=(34.0, 31.0, 34.1, 31.1),
            from_date="2024-01-01",
            to_date="2024-01-03",
            output_dir=tmp_path,
            client_id="explicit-id",
            client_secret="explicit-secret",
        )

    assert not credential_file.exists()


def test_save_creds_false_does_not_store_explicit_credentials(
    monkeypatch, tmp_path, credential_file
):
    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: "token")
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="explicit-id",
        client_secret="explicit-secret",
        save_creds=False,
    )

    assert not credential_file.exists()


def test_incomplete_explicit_credentials_fall_back_to_stored_pair(
    monkeypatch, tmp_path, credential_file
):
    acquire.store_cdse_credentials("stored-id", "stored-secret")
    token_calls = []
    monkeypatch.setattr(
        acquire,
        "get_cdse_token",
        lambda *args: token_calls.append(args) or "token",
    )
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="ignored-incomplete-id",
    )

    assert token_calls == [("stored-id", "stored-secret")]


def test_token_error_does_not_expose_secret(monkeypatch):
    class FailedResponse:
        status_code = 401
        text = "server echoed super-secret"

        def raise_for_status(self):
            raise requests.HTTPError("raw failure")

    import requests

    monkeypatch.setattr(acquire.requests, "post", lambda *args, **kwargs: FailedResponse())
    with pytest.raises(requests.HTTPError) as exc_info:
        acquire.get_cdse_token("client-id", "super-secret")

    assert "super-secret" not in str(exc_info.value)


### Evalscripts

def test_load_evalscript_includes_scl():
    script = load_evalscript("SCL")
    assert 'bands: ["SCL"]' in script
    assert 'sampleType: "UINT8"' in script


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


@pytest.mark.parametrize("veg_index", ["NDVI", "SAVI", "MSAVI"])
def test_load_evalscript_masks_vi_with_roptram_scl_classes(veg_index):
    script = load_evalscript(veg_index, scl_mask=True)
    assert 'bands: ["B04", "B08", "SCL"]' in script
    assert "[2, 4, 5, 10].includes(sample.SCL)" in script
    assert "return [NaN]" in script


def test_load_evalscript_keeps_explicit_scl_override():
    script = load_evalscript("NDVI", scl_mask=True, scl_keep={7, 4})
    assert "[4, 7].includes(sample.SCL)" in script


@pytest.mark.parametrize("veg_index", ["CI", "BSCI"])
def test_load_evalscript_rejects_indices_without_roptram_cdse_scripts(veg_index):
    with pytest.raises(ValueError, match="Unknown script_name"):
        load_evalscript(veg_index)


### Acquisition workflow

@pytest.fixture
def acquisition_mocks(monkeypatch):
    scene = {
        "id": "S2_TEST_T36RXV",
        "properties": {
            "datetime": "2024-01-02T10:20:30Z",
            "eo:cloud_cover": 3,
            "s2:mgrs_tile": "36RXV",
        },
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
    assert [call["script_name"] for call in downloads] == ["NDVI", "STR", "BOA"]
    assert downloads[0]["scl_mask"] is True


@pytest.mark.parametrize("max_cloud", [-1, 101, float("inf"), float("nan"), "12"])
def test_acquisition_rejects_invalid_max_cloud(tmp_path, max_cloud):
    with pytest.raises(ValueError, match="max_cloud"):
        acquire.acquire_optram_inputs(
            aoi=(34.0, 31.0, 34.1, 31.1),
            from_date="2024-01-01",
            to_date="2024-01-03",
            output_dir=tmp_path,
            client_id="id",
            client_secret="secret",
            max_cloud=max_cloud,
        )


@pytest.mark.parametrize("veg_index", ["NDVI", "SAVI", "MSAVI"])
def test_acquisition_propagates_vegetation_index(
    tmp_path, veg_index, acquisition_mocks
):
    _, downloads = acquisition_mocks

    results = acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="id",
        client_secret="secret",
        veg_index=veg_index,
    )

    vi_download = downloads[0]
    assert vi_download["script_name"] == veg_index
    assert Path(vi_download["output_path"]).parent.name == veg_index
    assert Path(vi_download["output_path"]).name.startswith(f"{veg_index}_")
    assert results[veg_index] == [str(vi_download["output_path"])]
    assert results["scenes"][0][veg_index] == str(vi_download["output_path"])
    if veg_index == "NDVI":
        assert "NDVI" in results


@pytest.mark.parametrize("veg_index", ["CI", "BSCI"])
def test_acquisition_rejects_indices_without_cdse_scripts(tmp_path, veg_index):
    with pytest.raises(ValueError, match="supports NDVI, SAVI, or MSAVI"):
        acquire.acquire_optram_inputs(
            aoi=(34.0, 31.0, 34.1, 31.1),
            from_date="2024-01-01",
            to_date="2024-01-03",
            output_dir=tmp_path,
            client_id="id",
            client_secret="secret",
            veg_index=veg_index,
        )


def test_empty_acquisition_uses_selected_vi_key(monkeypatch, tmp_path):
    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: "token")
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [])

    results = acquire.acquire_optram_inputs(
        aoi=(34.0, 31.0, 34.1, 31.1),
        from_date="2024-01-01",
        to_date="2024-01-03",
        output_dir=tmp_path,
        client_id="id",
        client_secret="secret",
        veg_index="SAVI",
    )

    assert results == {
        "SAVI": [],
        "STR": [],
        "BOA": [],
        "SCL": [],
        "scenes": [],
    }
