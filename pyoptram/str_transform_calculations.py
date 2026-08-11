### STR Transform Calculations
# Calculates SWIR Transformed Reflectance (STR) from surface reflectance
# values and BOA rasters: STR = (1 - SWIR)^2 / (2 * SWIR).

from pathlib import Path

import numpy as np
import rasterio

from .options import _UNSET, get_optram_option


### Calculate STR from SWIR reflectance values scaled to 0-1.
def calculate_str(swir):
    """Calculate SWIR Transformed Reflectance from scaled SWIR values.

    Parameters
    ----------
    swir : array-like
        Surface-reflectance values on the 0--1 scale.

    Returns
    -------
    numpy.ndarray
        ``(1 - SWIR) ** 2 / (2 * SWIR)`` for positive SWIR values. Zero,
        negative, and invalid inputs produce ``NaN``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(swir > 0, ((1.0 - swir) ** 2) / (2.0 * swir), np.nan)


### Raster preparation

### Find BOA files and create the STR output folder.
def prepare_str_inputs(boa_dir, str_dir=None):
    boa_dir = Path(boa_dir)

    if not boa_dir.exists() or not boa_dir.is_dir():
        return None

    boa_list = sorted(path for path in boa_dir.glob("*.tif") if "BOA_" in path.name)

    if not boa_list:
        return None

    str_dir = boa_dir.parent / "STR" if str_dir is None else Path(str_dir)
    str_dir.mkdir(parents=True, exist_ok=True)

    return boa_list, str_dir


### Convert one BOA raster into one STR raster.
def process_boa_file(tif_path, str_dir, swir_band):
    with rasterio.open(tif_path) as src:
        if swir_band < 1 or swir_band > src.count:
            raise ValueError(
                f"swir_band={swir_band} is out of range for file {tif_path.name}"
            )

        profile = src.profile.copy()
        swir = src.read(swir_band).astype(np.float32) / 10000.0
        str_arr = calculate_str(swir).astype(np.float32)

        profile.update(dtype="float32", count=1, nodata=np.nan)

        out_path = str_dir / tif_path.name.replace("BOA", "STR")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(str_arr, 1)

    return str(out_path)


### Create STR rasters for every BOA raster in a folder.
def optram_calculate_str(boa_dir, str_dir=None, swir_band=_UNSET):
    """Create STR rasters from every ``BOA_*.tif`` in a directory.

    Parameters
    ----------
    boa_dir : path-like
        Directory containing multiband bottom-of-atmosphere GeoTIFFs.
    str_dir : path-like, optional
        Output directory. Defaults to a sibling ``STR`` directory.
    swir_band : {11, 12}, optional
        One-based BOA band to transform. Defaults to the current
        ``SWIR_band`` option, initially 11.

    Returns
    -------
    list of str or None
        Written STR paths, or ``None`` when the BOA directory or matching
        inputs do not exist.

    Notes
    -----
    Input DN values are divided by 10,000. Each output is a one-band
    float32 GeoTIFF whose filename replaces ``BOA`` with ``STR``; existing
    outputs are overwritten, matching rOPTRAM's batch behavior.

    Raises
    ------
    ValueError
        If ``swir_band`` is not 11 or 12, or is unavailable in an input.
    """
    if swir_band is _UNSET:
        swir_band = get_optram_option("SWIR_band")
    if swir_band not in (11, 12):
        raise ValueError("swir_band must be 11 or 12")
    prepared = prepare_str_inputs(boa_dir, str_dir)

    if prepared is None:
        return None

    boa_list, str_dir = prepared
    str_list = []

    for tif_path in boa_list:
        str_list.append(process_boa_file(tif_path, str_dir, swir_band))

    print(f"Prepared: {len(str_list)} STR files")
    return str_list
