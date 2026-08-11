### VI-STR Table-Assembly Tests
# Verifies raster pairing, pixel filtering, scene metadata, feature masks,
# table limits, and CSV output.

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from pyoptram import optram_ndvi_str

_TRANSFORM = from_origin(10.0, 20.0, 1.0, 1.0)


### Raster test helpers

def _write_single_band_tif(path, array, dtype="float32"):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": _TRANSFORM,
        "nodata": None,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


### Table construction and filtering

def test_optram_ndvi_str_builds_dataframe_and_filters_zero_str(tmp_path):
    # NDVI and STR rasters become one dataframe with invalid STR zero removed.
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)

    dataframe = optram_ndvi_str([ndvi_path], [str_path])

    assert len(dataframe) == 3
    assert (dataframe["STR"] > 0).all()
    assert list(dataframe.columns) == [
        "X",
        "Y",
        "TimestampUTC",
        "Month",
        "Tile",
        "NDVI",
        "STR",
        "source_index",
        "row",
        "col",
        "ndvi_path",
        "str_path",
    ]
    np.testing.assert_allclose(dataframe["NDVI"].to_numpy(), [0.2, 0.6, 0.8])
    np.testing.assert_allclose(dataframe["STR"].to_numpy(), [1.0, 2.0, 3.0])


def test_optram_ndvi_str_scl_mask_filters_clouds(tmp_path):
    # Pixels whose SCL class isn't in scl_keep are dropped.
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"
    scl_path = tmp_path / "SCL_test.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scl = np.array([[4, 3], [6, 9]], dtype=np.uint8)  # keep=4, cloud=3, keep=6, unused=9

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)
    _write_single_band_tif(scl_path, scl, dtype="uint8")

    dataframe = optram_ndvi_str([ndvi_path], [str_path], scl_paths=[scl_path])

    assert "SCL" in dataframe.columns
    assert len(dataframe) == 2
    np.testing.assert_allclose(sorted(dataframe["NDVI"].to_numpy()), [0.2, 0.6])
    np.testing.assert_array_equal(sorted(dataframe["SCL"].to_numpy()), [4, 6])


def test_optram_ndvi_str_features_filters_and_labels_pixels(tmp_path):
    # Only pixels inside a feature polygon are kept, tagged with Feature_ID.
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)

    # Raster spans x:[10,12], y:[18,20]; this polygon covers only the right
    # column of pixels (x in [11, 12]).
    features_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"ID": 7},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[11, 18], [12, 18], [12, 20], [11, 20], [11, 18]]],
                },
            }
        ],
    }

    dataframe = optram_ndvi_str(
        [ndvi_path],
        [str_path],
        features=features_geojson,
        feature_id_col="ID",
    )

    assert "Feature_ID" in dataframe.columns
    assert len(dataframe) == 2
    assert (dataframe["Feature_ID"] == 7).all()
    assert set(dataframe["col"]) == {1}


def test_optram_ndvi_str_scene_metadata_overrides_filename_parsing(tmp_path):
    # scene_metadata records are used instead of fragile filename regexes.
    ndvi_path = tmp_path / "NDVI_scene_without_timestamp.tif"
    str_path = tmp_path / "STR_scene_without_timestamp.tif"

    ndvi = np.array([[0.5]], dtype=np.float32)
    str_array = np.array([[2.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)

    scene_metadata = [
        {
            "NDVI": str(ndvi_path),
            "STR": str(str_path),
            "datetime": "2023-06-15T10:20:30Z",
            "tile": "36RTV",
        }
    ]

    dataframe = optram_ndvi_str(
        [ndvi_path], [str_path], scene_metadata=scene_metadata
    )

    assert dataframe.loc[0, "Month"] == 6
    assert dataframe.loc[0, "Tile"] == "36RTV"
    assert dataframe.loc[0, "TimestampUTC"] == pd.Timestamp("2023-06-15T10:20:30Z")


def test_optram_ndvi_str_max_tbl_size_caps_assembly(tmp_path):
    # max_tbl_size stops assembly once the cap is reached, mid-scene if needed.
    ndvi_path_1 = tmp_path / "NDVI_a.tif"
    str_path_1 = tmp_path / "STR_a.tif"
    ndvi_path_2 = tmp_path / "NDVI_b.tif"
    str_path_2 = tmp_path / "STR_b.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path_1, ndvi)
    _write_single_band_tif(str_path_1, str_array)
    _write_single_band_tif(ndvi_path_2, ndvi)
    _write_single_band_tif(str_path_2, str_array)

    dataframe = optram_ndvi_str(
        [ndvi_path_1, ndvi_path_2],
        [str_path_1, str_path_2],
        max_tbl_size=3,
    )

    assert len(dataframe) == 3
    assert set(dataframe["source_index"]) == {0}


def test_optram_ndvi_str_max_rows_downsamples_final_table(tmp_path):
    # max_rows randomly downsamples the fully assembled/filtered table.
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"

    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)

    dataframe = optram_ndvi_str(
        [ndvi_path], [str_path], max_rows=2, random_state=0
    )

    assert len(dataframe) == 2


### Scene matching and output

def test_optram_ndvi_str_pairs_files_by_scene_name_in_str_order(tmp_path):
    ndvi_a = tmp_path / "NDVI_a.tif"
    ndvi_b = tmp_path / "NDVI_b.tif"
    str_a = tmp_path / "STR_a.tif"
    str_b = tmp_path / "STR_b.tif"

    _write_single_band_tif(ndvi_a, np.array([[0.1]], dtype=np.float32))
    _write_single_band_tif(ndvi_b, np.array([[0.9]], dtype=np.float32))
    _write_single_band_tif(str_a, np.array([[1.0]], dtype=np.float32))
    _write_single_band_tif(str_b, np.array([[2.0]], dtype=np.float32))

    dataframe = optram_ndvi_str(
        ndvi_paths=[ndvi_a, ndvi_b],
        str_paths=[str_b, str_a],
    )

    np.testing.assert_allclose(dataframe["NDVI"], [0.9, 0.1])
    np.testing.assert_allclose(dataframe["STR"], [2.0, 1.0])
    assert dataframe["source_index"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("ndvi_names", "str_names", "message"),
    [
        (["NDVI_a.tif"], ["STR_b.tif"], "missing VI.*missing STR"),
        (["NDVI_a.tif", "SAVI_a.tif"], ["STR_a.tif"], "Cannot match VI files uniquely"),
    ],
)
def test_optram_ndvi_str_rejects_missing_or_ambiguous_pairs(
    tmp_path, ndvi_names, str_names, message
):
    ndvi_paths = []
    for name in ndvi_names:
        path = tmp_path / name
        _write_single_band_tif(path, np.array([[0.5]], dtype=np.float32))
        ndvi_paths.append(path)

    str_paths = []
    for name in str_names:
        path = tmp_path / name
        _write_single_band_tif(path, np.array([[2.0]], dtype=np.float32))
        str_paths.append(path)

    with pytest.raises(ValueError, match=message):
        optram_ndvi_str(ndvi_paths, str_paths)


def test_optram_ndvi_str_rejects_unmatched_scl_scene(tmp_path):
    ndvi_path = tmp_path / "NDVI_a.tif"
    str_path = tmp_path / "STR_a.tif"
    scl_path = tmp_path / "SCL_b.tif"
    _write_single_band_tif(ndvi_path, np.array([[0.5]], dtype=np.float32))
    _write_single_band_tif(str_path, np.array([[2.0]], dtype=np.float32))
    _write_single_band_tif(scl_path, np.array([[4]], dtype=np.uint8), dtype="uint8")

    with pytest.raises(ValueError, match="missing SCL.*no VI/STR scene"):
        optram_ndvi_str([ndvi_path], [str_path], scl_paths=[scl_path])


def test_optram_ndvi_str_parses_roptram_date_and_tile_filename(tmp_path):
    ndvi_path = tmp_path / "NDVI_2022-11-11_T36RXV.tif"
    str_path = tmp_path / "STR_2022-11-11_T36RXV.tif"
    _write_single_band_tif(ndvi_path, np.array([[0.5]], dtype=np.float32))
    _write_single_band_tif(str_path, np.array([[2.0]], dtype=np.float32))

    dataframe = optram_ndvi_str([ndvi_path], [str_path])

    assert dataframe.loc[0, "TimestampUTC"] == pd.Timestamp("2022-11-11T00:00:00Z")
    assert dataframe.loc[0, "Month"] == 11
    assert dataframe.loc[0, "Tile"] == "36RXV"


def test_optram_ndvi_str_writes_requested_csv_and_creates_parent(tmp_path):
    ndvi_path = tmp_path / "NDVI_a.tif"
    str_path = tmp_path / "STR_a.tif"
    output_csv = tmp_path / "tables" / "vi_str.csv"
    _write_single_band_tif(ndvi_path, np.array([[0.5]], dtype=np.float32))
    _write_single_band_tif(str_path, np.array([[2.0]], dtype=np.float32))

    dataframe = optram_ndvi_str([ndvi_path], [str_path], output_csv=output_csv)

    assert output_csv.is_file()
    persisted = pd.read_csv(output_csv)
    assert list(persisted.columns) == list(dataframe.columns)
    np.testing.assert_allclose(persisted["NDVI"], dataframe["NDVI"])
    np.testing.assert_allclose(persisted["STR"], dataframe["STR"])
