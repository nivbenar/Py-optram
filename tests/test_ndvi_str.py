### VI-STR Table-Assembly Tests
# Verifies raster pairing, pixel filtering, scene metadata, feature masks,
# table limits, and CSV output.

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from pyoptram import optram_ndvi_str, optram_wetdry_coefficients

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


def _write_vi_str_pair(tmp_path, ndvi=None, str_array=None):
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"
    if ndvi is None:
        ndvi = np.full((2, 2), 0.5, dtype=np.float32)
    if str_array is None:
        str_array = np.full((2, 2), 2.0, dtype=np.float32)
    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array)
    return ndvi_path, str_path


def _feature_collection(properties, coordinates):
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        }],
    }


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


def test_optram_ndvi_str_features_label_without_filtering_pixels(tmp_path):
    # Feature membership labels pixels without changing the VI-STR population.
    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ndvi_path, str_path = _write_vi_str_pair(tmp_path, ndvi, str_array)

    # Raster spans x:[10,12], y:[18,20]; this polygon touches only the right
    # column of pixels.
    features_geojson = _feature_collection(
        {"ID": 7},
        [[11.1, 18.1], [11.9, 18.1], [11.9, 19.9],
         [11.1, 19.9], [11.1, 18.1]],
    )

    dataframe = optram_ndvi_str(
        [ndvi_path],
        [str_path],
        features=features_geojson,
        plot_colors="features",
    )

    assert "Feature_ID" in dataframe.columns
    assert len(dataframe) == 4
    assert (dataframe.loc[dataframe["col"] == 1, "Feature_ID"] == 7).all()
    assert dataframe.loc[dataframe["col"] == 0, "Feature_ID"].isna().all()


def test_optram_ndvi_str_ignores_features_outside_feature_plot_mode(tmp_path):
    ndvi_path, str_path = _write_vi_str_pair(tmp_path)
    features = _feature_collection(
        {"ID": 7},
        [[11.1, 18.1], [11.9, 18.1], [11.9, 19.9],
         [11.1, 19.9], [11.1, 18.1]],
    )

    dataframe = optram_ndvi_str(
        [ndvi_path], [str_path], features=features, plot_colors="no"
    )

    assert len(dataframe) == 4
    assert "Feature_ID" not in dataframe.columns


def test_optram_ndvi_str_does_not_invent_missing_feature_ids(tmp_path):
    ndvi_path, str_path = _write_vi_str_pair(tmp_path)
    features = _feature_collection(
        {"ZONE": 7},
        [[11.1, 18.1], [11.9, 18.1], [11.9, 19.9],
         [11.1, 19.9], [11.1, 18.1]],
    )

    dataframe = optram_ndvi_str(
        [ndvi_path], [str_path], features=features, plot_colors="feature"
    )

    assert len(dataframe) == 4
    assert "Feature_ID" not in dataframe.columns


def test_optram_ndvi_str_feature_rasterization_uses_all_touched(tmp_path):
    ndvi_path, str_path = _write_vi_str_pair(tmp_path)
    features = _feature_collection(
        {"ID": 9},
        [[11.01, 19.90], [11.10, 19.90], [11.10, 19.99],
         [11.01, 19.99], [11.01, 19.90]],
    )

    dataframe = optram_ndvi_str(
        [ndvi_path], [str_path], features=features, plot_colors="feature"
    )

    assert (dataframe["Feature_ID"] == 9).sum() == 1
    assert dataframe["Feature_ID"].isna().sum() == 3


def test_feature_labels_do_not_change_fitted_coefficients(tmp_path):
    rng = np.random.default_rng(42)
    ndvi_path = tmp_path / "NDVI_test.tif"
    str_path = tmp_path / "STR_test.tif"
    ndvi = rng.uniform(0.05, 0.85, size=(100, 100)).astype(np.float32)
    dry = 0.2 + 0.5 * ndvi
    wet = 0.8 + 2.0 * ndvi
    str_array = dry + rng.uniform(size=ndvi.shape) * (wet - dry)
    _write_single_band_tif(ndvi_path, ndvi)
    _write_single_band_tif(str_path, str_array.astype(np.float32))
    features = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"ID": 3},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[60, -80], [110, -80], [110, 20],
                                 [60, 20], [60, -80]]],
            },
        }],
    }
    unlabeled = optram_ndvi_str([ndvi_path], [str_path], random_state=0)
    labeled = optram_ndvi_str(
        [ndvi_path],
        [str_path],
        features=features,
        plot_colors="features",
        random_state=0,
    )

    _, unlabeled_coeffs, _ = optram_wetdry_coefficients(
        unlabeled, vi_step=0.02, return_outputs=True
    )
    _, labeled_coeffs, _ = optram_wetdry_coefficients(
        labeled, vi_step=0.02, return_outputs=True
    )

    pd.testing.assert_frame_equal(unlabeled_coeffs, labeled_coeffs)


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


def test_optram_ndvi_str_max_tbl_size_samples_each_scene_evenly(tmp_path):
    # rOPTRAM divides max_tbl_size across scenes and samples each scene.
    ndvi_path_1 = tmp_path / "NDVI_a.tif"
    str_path_1 = tmp_path / "STR_a.tif"
    ndvi_path_2 = tmp_path / "NDVI_b.tif"
    str_path_2 = tmp_path / "STR_b.tif"

    ndvi = np.full((100, 120), 0.5, dtype=np.float32)
    str_array = np.full((100, 120), 2.0, dtype=np.float32)

    _write_single_band_tif(ndvi_path_1, ndvi)
    _write_single_band_tif(str_path_1, str_array)
    _write_single_band_tif(ndvi_path_2, ndvi)
    _write_single_band_tif(str_path_2, str_array)

    dataframe = optram_ndvi_str(
        [ndvi_path_1, ndvi_path_2],
        [str_path_1, str_path_2],
        max_tbl_size=10_000,
        random_state=0,
    )

    assert len(dataframe) == 10_000
    assert dataframe["source_index"].value_counts().to_dict() == {0: 5000, 1: 5000}


@pytest.mark.parametrize("max_tbl_size", [9999, 0, -1, np.inf, np.nan])
def test_optram_ndvi_str_rejects_invalid_roptram_table_size(tmp_path, max_tbl_size):
    ndvi_path = tmp_path / "NDVI_a.tif"
    str_path = tmp_path / "STR_a.tif"
    _write_single_band_tif(ndvi_path, np.array([[0.5]], dtype=np.float32))
    _write_single_band_tif(str_path, np.array([[2.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="at least 10000"):
        optram_ndvi_str([ndvi_path], [str_path], max_tbl_size=max_tbl_size)


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
