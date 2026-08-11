### OPTRAM Option Tests
# Verifies rOPTRAM-compatible defaults, validation, reset behavior, and use by
# implemented Python workflows.

import numpy as np
import pytest

from pyoptram import calculate_vi, optram_options


@pytest.fixture(autouse=True)
def reset_options():
    optram_options(reset=True, show_opts=False)
    yield
    optram_options(reset=True, show_opts=False)


### rOPTRAM defaults and mutation

def test_implemented_option_defaults_match_roptram_zzz():
    assert optram_options(show_opts=False) == {
        "veg_index": "NDVI",
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
        "porosity": 0.4,
    }


def test_optram_options_sets_and_resets_a_valid_value():
    optram_options("veg_index", "SAVI", show_opts=False)
    assert optram_options(show_opts=False)["veg_index"] == "SAVI"

    optram_options(reset=True, show_opts=False)
    assert optram_options(show_opts=False)["veg_index"] == "NDVI"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("veg_index", "EVI"),
        ("max_cloud", 101),
        ("vi_step", 0),
        ("vi_step", 0.021),
        ("trapezoid_method", "cubic"),
        ("SWIR_band", 10),
        ("max_tbl_size", 9999),
        ("rm.low.vi", 1),
        ("plot_colors", "rainbow"),
        ("feature_col", None),
        ("edge_points", 1),
        ("only_vi_str", 0),
        ("tileid", "T36RXV"),
        ("scm_mask", 1),
        ("overwrite", 1),
        ("porosity", 1.0),
    ],
)
def test_optram_options_rejects_invalid_values(name, value):
    with pytest.raises(ValueError, match="Incorrect value"):
        optram_options(name, value, show_opts=False)


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
