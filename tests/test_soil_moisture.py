import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pyoptram import (
    calculate_soil_moisture,
    optram_calculate_soil_moisture,
    optram_options,
)
from pyoptram.soil_moisture import _parse_coefficients


def test_imports_roptram_linear_csv_without_changing_values(tmp_path):
    values = [-0.154980123174297, 3.31268567825634, -0.278092260372729, 6.78590775191129]
    table = pd.DataFrame(
        [values],
        columns=["intercept_dry", "slope_dry", "intercept_wet", "slope_wet"],
    )
    path = tmp_path / "coefficients_lin.csv"
    table.to_csv(path, index=False)
    parsed = _parse_coefficients(path)
    assert parsed == {
        "method": "linear",
        "dry": {"intercept": values[0], "slope": values[1]},
        "wet": {"intercept": values[2], "slope": values[3]},
    }
def test_imports_roptram_polynomial_csv_without_changing_values(tmp_path):
    values = [
        1.77465928441002,
        5.14427128567816,
        0.663810916472301,
        3.6746990051156,
        10.5378396219567,
        0.329124081696222,
    ]
    table = pd.DataFrame(
        [values],
        columns=[
            "alpha_dry",
            "beta1_dry",
            "beta2_dry",
            "alpha_wet",
            "beta1_wet",
            "beta2_wet",
        ],
    )
    path = tmp_path / "coefficients_pol.csv"
    table.to_csv(path, index=False)
    parsed = _parse_coefficients(path)
    assert parsed == {
        "method": "polynomial",
        "dry": {"alpha": values[0], "beta_1": values[1], "beta_2": values[2]},
        "wet": {"alpha": values[3], "beta_1": values[4], "beta_2": values[5]},
    }
def test_public_soil_moisture_defaults_match_roptram():
    optram_options(reset=True, show_opts=False)
    assert optram_options(show_opts=False)["porosity"] == 0.4
def test_default_soil_moisture_equals_explicit_roptram_behavior_without_clipping():
    vi = np.array([0.0, 0.0], dtype=np.float32)
    str_array = np.array([-1.0, 3.0], dtype=np.float32)
    coefficients = {
        "method": "linear",
        "dry": {"intercept": 2.0, "slope": 0.0},
        "wet": {"intercept": 0.0, "slope": 0.0},
    }
    default = calculate_soil_moisture(vi, str_array, coefficients)
    explicit = calculate_soil_moisture(vi, str_array, coefficients, porosity=0.4, clip=False)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_allclose(default, [0.6, -0.2])
def test_soil_moisture_preserves_roptram_observable_na_porosity_behavior():
    result = calculate_soil_moisture(
        np.array([0.5]),
        np.array([1.0]),
        {
            "method": "linear",
            "dry": {"intercept": 2.0, "slope": 0.0},
            "wet": {"intercept": 0.0, "slope": 0.0},
        },
        porosity=np.nan,
    )
    assert np.isnan(result[0])


def test_soil_moisture_replaces_compact_str_prefix(tmp_path):
    suffix = "20240102T102030_T36RXV.tif"
    vi_path = tmp_path / f"NDVI_{suffix}"
    str_path = tmp_path / f"STR_{suffix}"
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1, 1),
    }
    for path, value in ((vi_path, 0.5), (str_path, 1.0)):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.array([[value]], dtype=np.float32), 1)

    outputs = optram_calculate_soil_moisture(
        [vi_path],
        [str_path],
        {
            "method": "linear",
            "dry": {"intercept": 2.0, "slope": 0.0},
            "wet": {"intercept": 0.0, "slope": 0.0},
        },
        tmp_path / "SM",
    )

    assert outputs[0].endswith(f"SM_{suffix}")
