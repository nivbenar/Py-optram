import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from pyoptram import optram_ndvi_str, optram_wetdry_coefficients
_TRANSFORM = from_origin(10.0, 20.0, 1.0, 1.0)
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
def _feature_geodataframe(properties, coordinates):
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        [properties], geometry=[Polygon(coordinates)], crs="EPSG:4326"
    )
def test_optram_ndvi_str_builds_dataframe_and_filters_zero_str(tmp_path):
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
def test_optram_ndvi_str_features_label_without_filtering_pixels(tmp_path):
    ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    str_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ndvi_path, str_path = _write_vi_str_pair(tmp_path, ndvi, str_array)
    features = _feature_geodataframe(
        {"ID": 7},
        [[11.1, 18.1], [11.9, 18.1], [11.9, 19.9],
         [11.1, 19.9], [11.1, 18.1]],
    )
    dataframe = optram_ndvi_str(
        [ndvi_path],
        [str_path],
        features=features,
        plot_colors="features",
    )
    assert "Feature_ID" in dataframe.columns
    assert len(dataframe) == 4
    assert (dataframe.loc[dataframe["col"] == 1, "Feature_ID"] == 7).all()
    assert dataframe.loc[dataframe["col"] == 0, "Feature_ID"].isna().all()
def test_optram_ndvi_str_max_tbl_size_samples_each_scene_evenly(tmp_path):
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
def test_optram_ndvi_str_matches_vi_files_in_str_order(tmp_path):
    ndvi_a = tmp_path / "NDVI_a.tif"
    ndvi_b = tmp_path / "NDVI_b.tif"
    ndvi_extra = tmp_path / "NDVI_orphan.tif"
    str_a = tmp_path / "STR_a.tif"
    str_b = tmp_path / "STR_b.tif"
    str_unmatched = tmp_path / "STR_unmatched.tif"
    _write_single_band_tif(ndvi_a, np.array([[0.1]], dtype=np.float32))
    _write_single_band_tif(ndvi_b, np.array([[0.9]], dtype=np.float32))
    _write_single_band_tif(ndvi_extra, np.array([[0.5]], dtype=np.float32))
    _write_single_band_tif(str_a, np.array([[1.0]], dtype=np.float32))
    _write_single_band_tif(str_b, np.array([[2.0]], dtype=np.float32))
    _write_single_band_tif(str_unmatched, np.array([[3.0]], dtype=np.float32))
    dataframe = optram_ndvi_str(
        ndvi_paths=[ndvi_a, ndvi_b, ndvi_extra],
        str_paths=[str_b, str_unmatched, str_a],
    )
    np.testing.assert_allclose(dataframe["NDVI"], [0.9, 0.1])
    np.testing.assert_allclose(dataframe["STR"], [2.0, 1.0])
def test_optram_ndvi_str_rejects_multiple_vi_matches(tmp_path):
    str_path = tmp_path / "STR_a.tif"
    ndvi_path = tmp_path / "NDVI_a.tif"
    other_vi_path = tmp_path / "SAVI_a.tif"
    array = np.ones((1, 1), dtype=np.float32)
    for path in (str_path, ndvi_path, other_vi_path):
        _write_single_band_tif(path, array)
    with pytest.raises(ValueError, match="More than one VI file matches"):
        optram_ndvi_str([ndvi_path, other_vi_path], [str_path])
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
