### Soil-Moisture Compatibility Tests
# Verifies rOPTRAM coefficient parsing and public soil-moisture defaults.

import numpy as np
import pandas as pd
import pytest

from pyoptram import (
    calculate_soil_moisture,
    optram_calculate_soil_moisture,
    optram_options,
)
from pyoptram.soil_moisture import _parse_coefficients


### rOPTRAM coefficient compatibility

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


def test_wide_four_column_dataframe_requires_explicit_linear_method():
    table = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        columns=["intercept_dry", "slope_dry", "intercept_wet", "slope_wet"],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _parse_coefficients(table)

    parsed = _parse_coefficients(table, method="linear")
    assert parsed["dry"] == {"intercept": 1.0, "slope": 2.0}
    assert parsed["wet"] == {"intercept": 3.0, "slope": 4.0}


def test_rejects_roptram_exponential_coefficient_file(tmp_path):
    path = tmp_path / "coefficients_exp.csv"
    pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        columns=["intercept_dry", "slope_dry", "intercept_wet", "slope_wet"],
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="exponential coefficient files are not supported"):
        _parse_coefficients(path)


### Public soil-moisture behavior

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


@pytest.mark.parametrize("porosity", [0, -0.1, 1, 1.1, np.inf])
def test_soil_moisture_rejects_porosity_outside_roptram_range(porosity):
    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        calculate_soil_moisture(
            np.array([0.5]),
            np.array([1.0]),
            {
                "method": "linear",
                "dry": {"intercept": 2.0, "slope": 0.0},
                "wet": {"intercept": 0.0, "slope": 0.0},
            },
            porosity=porosity,
        )


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
