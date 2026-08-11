### Vegetation-Index Calculations
# Calculates the NDVI, SAVI, MSAVI, CI, and BSCI indices implemented by
# pyOPTRAM from band-first raster arrays.

import numpy as np


SUPPORTED_VEGETATION_INDICES = ("NDVI", "SAVI", "MSAVI", "CI", "BSCI")


### Band scaling

def _scaled_band(img_stack, band_number, scale_factor):
    """Select and rescale one band using rOPTRAM's one-based numbering."""
    band_index = band_number - 1
    if band_number < 1 or band_index >= img_stack.shape[0]:
        raise ValueError(
            f"Band {band_number} is not available in a {img_stack.shape[0]}-band stack"
        )

    band = img_stack[band_index].astype(np.float64)
    if np.all(np.isnan(band)):
        band_minimum = np.nan
    else:
        band_minimum = np.nanmin(band)
    return 255 * (band - band_minimum) / scale_factor


### Calculate a selected vegetation index from scaled raster bands.
def calculate_vi(img_stack, veg_index="NDVI", redband=4, greenband=3, blueband=2,
                 nirband=5, scale_factor=2**15):
    """Calculate a vegetation index from a band-first raster array.

    Band numbers are one-based, as in rOPTRAM's ``calculate_vi`` function.
    Unlike the current R implementation, scaling is applied once rather than
    twice, and only the bands needed by the selected index are required. These
    are intentional corrections of apparent rOPTRAM implementation errors.
    """
    if veg_index not in SUPPORTED_VEGETATION_INDICES:
        raise ValueError("veg_index must be one of: " + ", ".join(SUPPORTED_VEGETATION_INDICES))

    if not np.isscalar(scale_factor) or not np.isfinite(scale_factor):
        raise ValueError("scale_factor must be a finite positive number")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be a finite positive number")

    img_stack = np.asarray(img_stack)
    if img_stack.ndim < 2:
        raise ValueError("img_stack must be a band-first array with at least 2 dimensions")

    required_bands = {
        "NDVI": {"red": redband, "nir": nirband},
        "SAVI": {"red": redband, "nir": nirband},
        "MSAVI": {"red": redband, "nir": nirband},
        "CI": {"red": redband, "blue": blueband},
        "BSCI": {"red": redband, "green": greenband, "nir": nirband},
    }[veg_index]

    bands = {name: _scaled_band(img_stack, number, scale_factor)
             for name, number in required_bands.items()}

    with np.errstate(divide="ignore", invalid="ignore"):
        if veg_index == "NDVI":
            vi_array = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])
        elif veg_index == "SAVI":
            vi_array = 1.5 * (bands["nir"] - bands["red"]) / (
                bands["nir"] + bands["red"] + 0.5
            )
        elif veg_index == "MSAVI":
            vi_array = (
                2 * bands["nir"]
                + 1
                - np.sqrt(
                    (2 * bands["nir"] + 1) ** 2
                    - 8 * (bands["nir"] - bands["red"])
                )
            ) / 2
        elif veg_index == "CI":
            vi_array = 1 - (bands["red"] - bands["blue"]) / (bands["red"] + bands["blue"])
        else:
            mean_band = np.nanmean(
                np.stack([bands["green"], bands["red"], bands["nir"]]),
                axis=0,
            )
            vi_array = (1 - 2 * (bands["red"] - bands["green"])) / mean_band

    return vi_array
