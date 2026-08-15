import numpy as np
import pytest
from pyoptram import calculate_vi
def _scale(band, scale_factor=2**15):
    return 255 * (band - np.nanmin(band)) / scale_factor
def _stack():
    return np.array(
        [
            [[5.0, 10.0], [15.0, 20.0]],
            [[2.0, 5.0], [8.0, 11.0]],
            [[3.0, 7.0], [10.0, 15.0]],
            [[4.0, 9.0], [13.0, 18.0]],
            [[8.0, 15.0], [19.0, 25.0]],
        ]
    )
@pytest.mark.parametrize("veg_index", ["NDVI", "SAVI", "MSAVI", "CI", "BSCI"])
def test_calculate_vi_matches_roptram_formulas(veg_index):
    stack = _stack()
    blue = _scale(stack[1])
    green = _scale(stack[2])
    red = _scale(stack[3])
    nir = _scale(stack[4])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = {
            "NDVI": (nir - red) / (nir + red),
            "SAVI": 1.5 * (nir - red) / (nir + red + 0.5),
            "MSAVI": (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2,
            "CI": 1 - (red - blue) / (red + blue),
            "BSCI": ((1 - 2 * (red - green))
                     / np.nanmean(np.stack([green, red, nir]), axis=0)),
        }[veg_index]
    result = calculate_vi(stack, veg_index=veg_index)
    np.testing.assert_allclose(result, expected, equal_nan=True)
def test_calculate_vi_applies_approved_single_scaling_pass():
    stack = _stack()
    red_once = _scale(stack[3])
    nir_once = _scale(stack[4])
    expected_once = 1.5 * (nir_once - red_once) / (nir_once + red_once + 0.5)
    red_twice = _scale(red_once)
    nir_twice = _scale(nir_once)
    accidental_roptram_result = 1.5 * (nir_twice - red_twice) / (nir_twice + red_twice + 0.5)
    result = calculate_vi(stack, veg_index="SAVI")
    np.testing.assert_allclose(result, expected_once, equal_nan=True)
    assert not np.allclose(result, accidental_roptram_result, equal_nan=True)
def test_calculate_vi_uses_one_based_custom_bands_and_scale_factor():
    stack = np.array(
        [
            [[10.0, 20.0, 30.0]],
            [[20.0, 40.0, 80.0]],
        ]
    )
    red = _scale(stack[0], scale_factor=255)
    nir = _scale(stack[1], scale_factor=255)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (nir - red) / (nir + red)
    result = calculate_vi(stack, redband=1, nirband=2, scale_factor=255)
    np.testing.assert_allclose(result, expected, equal_nan=True)
def test_calculate_vi_returns_nan_for_zero_denominator():
    stack = np.array(
        [
            [[1.0, 2.0]],
            [[1.0, 2.0]],
        ]
    )
    result = calculate_vi(stack, redband=1, nirband=2)
    assert np.isnan(result[0, 0])
    assert result[0, 1] == 0
def test_calculate_vi_rejects_missing_required_band():
    with pytest.raises(ValueError, match="is not available"):
        calculate_vi(np.ones((2, 3, 3)))
