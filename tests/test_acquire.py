from pathlib import Path

import pytest

import pyoptram.acquire as acquire
from pyoptram.acquire import load_evalscript


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


@pytest.mark.parametrize("veg_index", ["NDVI", "SAVI", "MSAVI"])
def test_acquisition_propagates_vegetation_index(monkeypatch, tmp_path, veg_index):
    scene = {
        "id": "S2_TEST_T36RXV",
        "properties": {
            "datetime": "2024-01-02T10:20:30Z",
            "eo:cloud_cover": 3,
            "s2:mgrs_tile": "36RXV",
        },
    }
    downloads = []

    monkeypatch.setattr(acquire, "get_cdse_token", lambda *args: "token")
    monkeypatch.setattr(acquire, "search_catalog", lambda **kwargs: [scene])

    def fake_download_index(**kwargs):
        downloads.append(kwargs)
        return str(kwargs["output_path"])

    monkeypatch.setattr(acquire, "download_index", fake_download_index)

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
