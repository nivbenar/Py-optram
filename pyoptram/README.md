# pyoptram

`pyoptram` is a Python implementation of core OPTRAM workflow steps inspired by [rOPTRAM](https://github.com/ropensci/rOPTRAM).

It currently focuses on:
- acquiring Sentinel-2 NDVI and STR inputs from Copernicus Data Space,
- preparing paired NDVI-STR pixel tables,
- fitting OPTRAM wet/dry edge coefficients,
- calculating soil moisture rasters from fitted coefficients,
- plotting VI-STR clouds.

## Install (editable)

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[geo]
```

## Quick start

```python
from pyoptram import (
    acquire_optram_inputs,
    optram_calculate_soil_moisture,
    optram_ndvi_str,
    optram_wetdry_coefficients,
    store_cdse_credentials,
    store_cdse_credentials_from_file,
)

# One-time setup. OAUTH_CLIENTID and OAUTH_SECRET may be used instead.
store_cdse_credentials(
    client_id="YOUR_CDSE_CLIENT_ID",
    client_secret="YOUR_CDSE_CLIENT_SECRET",
)

# Or initialize from a CSV with clientid,secret headers:
# store_cdse_credentials_from_file("path/to/cdse_credentials.csv")

acquired = acquire_optram_inputs(
    aoi="path/to/aoi.geojson",  # GeoJSON, vector file, dict, or bbox tuple
    from_date="2024-01-01",
    to_date="2024-03-31",
    output_dir="data/optram",
    max_cloud=20,
    only_vi_str=True,
    download_scl=True,
)

df = optram_ndvi_str(
    ndvi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    scl_paths=acquired["SCL"],
    rm_low_vi=False,
    rm_hi_str=False,
)

rmse_df, coeffs_df, edges_df = optram_wetdry_coefficients(
    full_df=df,
    method="linear",
    return_outputs=True,
)

sm_paths = optram_calculate_soil_moisture(
    vi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    coefficients=coeffs_df,
    output_dir="data/optram/SM",
    porosity=0.4,
)
```

Soil-moisture calculations default to rOPTRAM's `porosity=0.4` and do not
clip values. Pass `clip=True` explicitly when bounded output is desired.

## rOPTRAM-like `optram_ndvi_str` options

The NDVI/STR table builder now supports quality masking, feature extraction, and size caps:

```python
features_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"ID": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[34.8, 31.4], [34.9, 31.4], [34.9, 31.5], [34.8, 31.5], [34.8, 31.4]]],
            },
        }
    ],
}

df = optram_ndvi_str(
    ndvi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    scl_paths=acquired["SCL"],        # optional SCL cloud-quality mask input
    scl_keep={4, 5, 6, 7},            # keep SCL classes (defaults to {4,5,6,7})
    features=features_geojson,        # optional polygon filtering
    feature_id_col="ID",              # creates Feature_ID column
    max_tbl_size=1_000_000,           # hard cap while assembling
    max_rows=250_000,                 # optional final downsample
    scene_metadata=acquired["scenes"] # robust datetime/tile metadata lookup
)
```

VI, STR, and optional SCL files are paired by the filename portion after the
product prefix (for example, `NDVI_2022-11-11_T36RXV.tif` pairs with
`STR_2022-11-11_T36RXV.tif`). Pairing is deterministic and follows STR input
order. Missing or duplicate scene products raise `ValueError`; the function
never silently runs a partial set of scenes. Use `output_csv` to persist the
returned table to a caller-selected CSV path.

### VI–STR compatibility notes

The Python workflow intentionally retains its existing behavior where it
differs from rOPTRAM:

- Python returns `X`, `Y`, and `NDVI` plus pixel/source provenance columns;
  rOPTRAM's implementation returns lowercase `x`, `y`, and generic `VI`.
- Python requires identical VI/STR grids. rOPTRAM joins raster values by
  coordinates and can therefore create a partial intersection.
- Python always filters non-finite values, VI outside `[-1, 1]`, and
  non-positive STR. Its optional low-VI, high-STR, SCL, and feature filtering
  behavior is unchanged.
- Python's `max_tbl_size` is an optional row-order assembly cap and `max_rows`
  is an optional final random sample. rOPTRAM distributes its table-size cap
  across scenes and randomly samples each oversized scene.
- Python writes a caller-named CSV only when `output_csv` is supplied;
  rOPTRAM always writes `VI_STR_data.rds` to its output directory.
- Python returns an empty dataframe with a stable schema when matched scenes
  contain no valid pixels. rOPTRAM generally skips empty scenes and uses
  `NULL` for several empty-input cases.
- Python feature input filters pixels to the supplied geometries and adds
  `Feature_ID`; rOPTRAM's AOI path is intended primarily to label features for
  plot coloring.

These are documented compatibility differences. This milestone does not
change scientific defaults, formulas, filtering thresholds, mask semantics,
sampling behavior, or calculation order.

## Available API

- `get_cdse_token`
- `acquire_optram_inputs`
- `calculate_str`
- `optram_calculate_str`
- `optram_ndvi_str`
- `optram_wetdry_coefficients`
- `calculate_soil_moisture`
- `optram_calculate_soil_moisture`
- `plot_vi_str_cloud`

## Current status

This package is functional for coefficient generation, but still under active development toward fuller feature parity with rOPTRAM.

Planned additions include:
- a one-call `optram(...)` wrapper,
- broader documentation and tests.
