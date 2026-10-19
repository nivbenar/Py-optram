import inspect

import numpy as np
import pytest

import pyoptram
from pyoptram import calculate_vi, optram_ndvi_str, optram_options


@pytest.fixture(autouse=True)
def reset_options():
    optram_options(reset=True, show_opts=False)
    yield
    optram_options(reset=True, show_opts=False)


def test_public_signatures_display_current_option_for_sentinel_defaults():
    signatures = {
        name: str(inspect.signature(getattr(pyoptram, name)))
        for name in pyoptram.__all__
    }

    assert all("<object object at" not in signature for signature in signatures.values())

    ndvi_str_parameters = inspect.signature(optram_ndvi_str).parameters
    for name in (
        "rm_low_vi",
        "rm_hi_str",
        "feature_id_col",
        "plot_colors",
        "max_tbl_size",
    ):
        assert repr(ndvi_str_parameters[name].default) == "CURRENT_OPTION"


def test_implemented_option_defaults_match_roptram_zzz():
    assert optram_options(show_opts=False) == {
        "veg_index": "NDVI",
        "period": "full",
        "max_cloud": 12,
        "vi_step": 0.005,
        "trapezoid_method": "linear",
        "SWIR_band": 11,
        "max_tbl_size": 1_000_000,
        "rm.low.vi": False,
        "rm.hi.str": False,
        "plot_colors": "no",
        "feature_col": "ID",
        "edge_points": True,
        "only_vi_str": False,
        "tileid": None,
        "scm_mask": True,
        "overwrite": False,
        "save_img_list": False,
        "resolution": 10,
        "area_cover": 99.0,
        "porosity": 0.4,
    }
def test_optram_options_sets_and_resets_a_valid_value():
    optram_options("veg_index", "SAVI", show_opts=False)
    assert optram_options(show_opts=False)["veg_index"] == "SAVI"
    optram_options(reset=True, show_opts=False)
    assert optram_options(show_opts=False)["veg_index"] == "NDVI"


def test_optram_options_display_is_opt_in(capsys):
    optram_options()
    assert capsys.readouterr().out == ""

    optram_options(show_opts=True)
    output = capsys.readouterr().out
    assert "veg_index = NDVI\n" in output
    assert "porosity = 0.4\n" in output


@pytest.mark.parametrize("period", ["full", "seasonal"])
def test_optram_options_accepts_roptram_periods(period):
    optram_options("period", period, show_opts=False)
    assert optram_options(show_opts=False)["period"] == period
def test_configured_vegetation_index_is_used_when_argument_is_omitted():
    stack = np.array(
        [
            [[1.0, 2.0]],
            [[2.0, 3.0]],
            [[3.0, 4.0]],
            [[4.0, 5.0]],
            [[8.0, 10.0]],
        ]
    )
    optram_options("veg_index", "SAVI", show_opts=False)
    configured = calculate_vi(stack)
    explicit_savi = calculate_vi(stack, veg_index="SAVI")
    explicit_ndvi = calculate_vi(stack, veg_index="NDVI")

    np.testing.assert_allclose(configured, explicit_savi, equal_nan=True)
    assert not np.allclose(configured, explicit_ndvi, equal_nan=True)


def test_explicit_falsey_and_none_option_values_are_not_treated_as_omitted():
    optram_options("rm.low.vi", True, show_opts=False)
    optram_options("rm.low.vi", False, show_opts=False)
    assert optram_options(show_opts=False)["rm.low.vi"] is False

    optram_options("tileid", "32TQM", show_opts=False)
    optram_options("tileid", None, show_opts=False)
    assert optram_options(show_opts=False)["tileid"] is None
