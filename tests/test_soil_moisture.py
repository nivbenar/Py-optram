import pandas as pd
import pytest

from pyoptram.soil_moisture import _parse_coefficients


def test_imports_roptram_linear_csv_without_changing_values(tmp_path):
    values = [-0.154980123174297, 3.31268567825634, -0.278092260372729, 6.78590775191129]
    table = pd.DataFrame(
        [values],
        columns=[
            "intercept_dry",
            "slope_dry",
            "intercept_wet",
            "slope_wet",
        ],
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
        columns=[
            "intercept_dry",
            "slope_dry",
            "intercept_wet",
            "slope_wet",
        ],
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
        columns=[
            "intercept_dry",
            "slope_dry",
            "intercept_wet",
            "slope_wet",
        ],
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="exponential coefficient files are not supported"):
        _parse_coefficients(path)
