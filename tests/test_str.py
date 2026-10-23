import numpy as np
import rasterio
from rasterio.transform import from_origin

from pyoptram import calculate_str
from pyoptram.str_transform_calculations import process_boa_file


def test_calculate_str_formula_and_non_positive_values():
    swir = np.array([0.1, 0.2, 0.0, -0.1], dtype=np.float32)
    result = calculate_str(swir)
    expected = np.array([4.05, 1.6, np.nan, np.nan], dtype=np.float32)
    np.testing.assert_allclose(result[:2], expected[:2], rtol=1e-6)
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_process_boa_file_preserves_declared_nodata(tmp_path):
    boa_path = tmp_path / "BOA_20240102T102030_T36RXV.tif"
    str_dir = tmp_path / "STR"
    str_dir.mkdir()

    boa = np.zeros((11, 1, 2), dtype=np.uint16)
    boa[10, 0] = [1000, 65535]
    with rasterio.open(
        boa_path,
        "w",
        driver="GTiff",
        height=1,
        width=2,
        count=11,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1, 1),
        nodata=65535,
    ) as dst:
        dst.write(boa)

    output_path = process_boa_file(boa_path, str_dir, swir_band=11)

    assert output_path.endswith("STR_20240102T102030_T36RXV.tif")

    with rasterio.open(output_path) as src:
        result = src.read(1)
        assert src.dtypes == ("float32",)
        assert np.isnan(src.nodata)

    np.testing.assert_allclose(result[0, 0], 4.05, rtol=1e-6)
    assert np.isnan(result[0, 1])
