from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest

wrapper = import_module("pyoptram.optram")


def _image_directories(base_dir):
    (base_dir / "NDVI").mkdir(parents=True)
    (base_dir / "STR").mkdir()


def test_optram_fits_once_plots_edges_closes_figure_and_returns_rmse(
    monkeypatch, tmp_path
):
    s2_dir = tmp_path / "s2"
    data_dir = tmp_path / "data"
    _image_directories(s2_dir)
    vi_str = object()
    edges = object()
    rmse = object()
    figure = object()
    fit_calls = 0
    plotted = {}
    closed = []

    monkeypatch.setattr(wrapper, "acquire_optram_inputs", lambda **kwargs: None)
    monkeypatch.setattr(wrapper, "optram_ndvi_str", lambda *args, **kwargs: vi_str)

    def fit(full_df, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        assert full_df is vi_str
        assert kwargs == {"output_dir": data_dir, "return_outputs": True}
        return rmse, object(), edges

    def plot(full_df, edges_df, **kwargs):
        plotted.update(full_df=full_df, edges_df=edges_df, kwargs=kwargs)
        return SimpleNamespace(figure=figure)

    monkeypatch.setattr(wrapper, "optram_wetdry_coefficients", fit)
    monkeypatch.setattr(wrapper, "plot_vi_str_cloud", plot)
    monkeypatch.setattr(wrapper.plt, "close", closed.append)

    result = wrapper.optram(
        "aoi", "2024-01-01", "2024-01-03", s2_dir, data_dir
    )

    assert fit_calls == 1
    assert plotted == {
        "full_df": vi_str,
        "edges_df": edges,
        "kwargs": {"output_path": data_dir / "vi_str_trapezoid.png"},
    }
    assert closed == [figure]
    assert result is rmse


def test_optram_creates_real_non_empty_plot(monkeypatch, tmp_path):
    s2_dir = tmp_path / "s2"
    data_dir = tmp_path / "results"
    _image_directories(s2_dir)
    vi_str = pd.DataFrame({"VI": [0.1, 0.2], "STR": [1.0, 1.2]})
    edges = pd.DataFrame(
        {
            "VI": [0.1, 0.2],
            "STR_wet": [1.1, 1.3],
            "STR_dry": [0.8, 0.9],
            "STR_wet_fit": [1.1, 1.3],
            "STR_dry_fit": [0.8, 0.9],
        }
    )
    rmse = pd.DataFrame([{"RMSE wet": 0.0, "RMSE dry": 0.0}])

    monkeypatch.setattr(wrapper, "acquire_optram_inputs", lambda **kwargs: None)
    monkeypatch.setattr(wrapper, "optram_ndvi_str", lambda *args, **kwargs: vi_str)
    monkeypatch.setattr(
        wrapper,
        "optram_wetdry_coefficients",
        lambda *args, **kwargs: (rmse, object(), edges),
    )

    wrapper.optram("aoi", "2024-01-01", "2024-01-03", s2_dir, data_dir)

    output_path = data_dir / "vi_str_trapezoid.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_optram_propagates_plot_failure_without_refitting(monkeypatch, tmp_path):
    s2_dir = tmp_path / "s2"
    _image_directories(s2_dir)
    fit_calls = 0

    monkeypatch.setattr(wrapper, "acquire_optram_inputs", lambda **kwargs: None)
    monkeypatch.setattr(wrapper, "optram_ndvi_str", lambda *args, **kwargs: "table")

    def fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return "rmse", "coefficients", "edges"

    def fail_plot(*args, **kwargs):
        raise OSError("cannot save plot")

    monkeypatch.setattr(wrapper, "optram_wetdry_coefficients", fit)
    monkeypatch.setattr(wrapper, "plot_vi_str_cloud", fail_plot)

    with pytest.raises(OSError, match="cannot save plot"):
        wrapper.optram(
            "aoi",
            "2024-01-01",
            "2024-01-03",
            s2_dir,
            tmp_path / "data",
        )

    assert fit_calls == 1
