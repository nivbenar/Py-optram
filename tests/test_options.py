import numpy as np
import pytest
from pyoptram import calculate_vi, optram_options
@pytest.fixture(autouse=True)
def reset_options():
    optram_options(reset=True, show_opts=False)
    yield
    optram_options(reset=True, show_opts=False)
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
    explicit = calculate_vi(stack, veg_index="SAVI")
    np.testing.assert_allclose(configured, explicit, equal_nan=True)
