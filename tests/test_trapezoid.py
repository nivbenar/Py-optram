import numpy as np
import pandas as pd

from pyoptram import optram_wetdry_coefficients
from pyoptram.soil_moisture import _parse_coefficients


def _fitting_dataframe():
    rng = np.random.default_rng(42)
    ndvi = np.repeat(np.linspace(0.05, 0.85, 17), 80)
    wet_line = 0.8 + 2.0 * ndvi
    dry_line = 0.2 + 0.5 * ndvi
    position = rng.uniform(0.0, 1.0, size=ndvi.size)
    str_values = dry_line + position * (wet_line - dry_line)
    return pd.DataFrame({"NDVI": ndvi, "STR": str_values})


def test_optram_wetdry_coefficients_returns_rmse_coefficients_and_edges():
    # Wet/dry fitting returns RMSE, coefficients, and fitted edge points.
    dataframe = _fitting_dataframe()

    rmse_df, coeffs_df, edges_df = optram_wetdry_coefficients(
        dataframe,
        method="linear",
        vi_step=0.1,
        min_bin_count=20,
        return_outputs=True,
    )

    assert list(rmse_df.columns) == ["RMSE wet", "RMSE dry"]
    assert set(coeffs_df["edge"]) == {"wet", "dry"}
    assert {"STR_wet", "STR_dry", "STR_wet_fit", "STR_dry_fit"}.issubset(
        edges_df.columns
    )
    assert rmse_df.loc[0, "RMSE wet"] >= 0
    assert rmse_df.loc[0, "RMSE dry"] >= 0


def test_exports_and_round_trips_roptram_linear_coefficients_exactly(tmp_path):
    _, coeffs_df, _ = optram_wetdry_coefficients(
        _fitting_dataframe(),
        output_dir=tmp_path,
        method="linear",
        vi_step=0.1,
        min_bin_count=20,
        return_outputs=True,
        export_roptram=True,
    )

    path = tmp_path / "coefficients_lin.csv"
    exported = pd.read_csv(path)
    assert list(exported.columns) == [
        "intercept_dry",
        "slope_dry",
        "intercept_wet",
        "slope_wet",
    ]

    parsed = _parse_coefficients(path)
    wet = coeffs_df[coeffs_df["edge"] == "wet"].iloc[0]
    dry = coeffs_df[coeffs_df["edge"] == "dry"].iloc[0]
    assert parsed["dry"] == {
        "intercept": dry["intercept"],
        "slope": dry["slope"],
    }
    assert parsed["wet"] == {
        "intercept": wet["intercept"],
        "slope": wet["slope"],
    }
    assert (tmp_path / "wetdry_coefficients.csv").is_file()
    assert (tmp_path / "wetdry_rmse.csv").is_file()
    assert (tmp_path / "trapezoid_points.csv").is_file()


def test_exports_and_round_trips_roptram_polynomial_coefficients_exactly(tmp_path):
    _, coeffs_df, _ = optram_wetdry_coefficients(
        _fitting_dataframe(),
        output_dir=tmp_path,
        method="polynomial",
        vi_step=0.1,
        min_bin_count=20,
        return_outputs=True,
        export_roptram=True,
    )

    path = tmp_path / "coefficients_pol.csv"
    exported = pd.read_csv(path)
    assert list(exported.columns) == [
        "alpha_dry",
        "beta1_dry",
        "beta2_dry",
        "alpha_wet",
        "beta1_wet",
        "beta2_wet",
    ]

    parsed = _parse_coefficients(path)
    wet = coeffs_df[coeffs_df["edge"] == "wet"].iloc[0]
    dry = coeffs_df[coeffs_df["edge"] == "dry"].iloc[0]
    assert parsed["dry"] == {
        "alpha": dry["alpha"],
        "beta_1": dry["beta_1"],
        "beta_2": dry["beta_2"],
    }
    assert parsed["wet"] == {
        "alpha": wet["alpha"],
        "beta_1": wet["beta_1"],
        "beta_2": wet["beta_2"],
    }


def test_roptram_compatibility_export_is_opt_in(tmp_path):
    optram_wetdry_coefficients(
        _fitting_dataframe(),
        output_dir=tmp_path,
        method="linear",
        vi_step=0.1,
        min_bin_count=20,
    )

    assert not (tmp_path / "coefficients_lin.csv").exists()
