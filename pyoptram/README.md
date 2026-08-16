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
    optram,
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

# Thin counterpart to rOPTRAM::optram() for the supported CDSE path.
rmse = optram(
    aoi="path/to/aoi.geojson",
    from_date="2024-01-01",
    to_date="2024-03-31",
    s2_output_dir="data/optram",
    data_output_dir="data/optram/results",
)

# optram() acquires VI/STR imagery, creates the VI-STR table, fits wet/dry
# edges, and prints and returns RMSE. Matching rOPTRAM::optram(), it does not
# calculate soil moisture. The current wrapper supports CDSE; openEO backend
# parity remains future work.

acquired = acquire_optram_inputs(
    aoi="path/to/aoi.geojson",  # GeoJSON, vector file, dict, or bbox tuple
    from_date="2024-01-01",
    to_date="2024-03-31",
    output_dir="data/optram",
    max_cloud=12,
    area_cover=99.0,
    period="full",
    save_img_list=False,
    only_vi_str=False,
    resolution=10,
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

For acquisition, multi-feature AOIs are geometrically unioned before catalog
search and download, matching rOPTRAM. Date ranges must satisfy
`to_date > from_date`. Acquisition resolution defaults to 10 metres and accepts
10, 20, or 60 metres. Matching rOPTRAM and `CDSE::GetImage()`, metre resolution
is converted to a CRS84 angular grid at the AOI latitude. Explicit `width` and
`height` remain available as Python-specific Process API overrides.

Catalog search consumes every CDSE page by default. Following rOPTRAM, scenes
are filtered client-side in this order: strict cloud cover (`< max_cloud`),
case-sensitive tile substring in the catalog `sourceId`, spherical AOI coverage
rounded to three decimals (`>= area_cover`), and then the optional seasonal
date filter. `period="seasonal"` repeats the month/day window from the supplied
dates in each covered year and returns scenes in descending acquisition-date
order. Process requests use each selected acquisition's whole UTC day with
`mosaickingOrder="mostRecent"`.

When `save_img_list=True`, the post-filter scene list is written before any
downloads to `image_list.json` as a GeoJSON FeatureCollection. rOPTRAM writes
the corresponding R object as `image_list.rds`; genuine RDS writing is not
available from Py-optram's approved runtime dependencies. Spherical coverage
uses `s2rst` rather than planar Shapely. Tiny differences from R sf/S2 can occur
at artificial floating-point rounding boundaries; practical parity after
three-decimal rounding is the supported policy.

## rOPTRAM-compatible package options

Implemented workflow options use rOPTRAM names, defaults, and validation:

```python
from pyoptram import optram_options

optram_options("vi_step", 0.01, show_opts=False)
optram_options("plot_colors", "density", show_opts=False)
current_options = optram_options(show_opts=False)
```

Explicit function arguments override the corresponding session option.

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
    features=features_geojson,        # optional feature labels for plotting
    feature_id_col="ID",              # creates Feature_ID column
    plot_colors="features",           # enables feature-ID preparation
    max_tbl_size=1_000_000,           # divided and sampled across STR files
    max_rows=250_000,                 # optional final downsample
)
```

Each STR file is matched to a VI file by its filename identifier (for example,
`STR_2022-11-11_T36RXV.tif` matches `NDVI_2022-11-11_T36RXV.tif`). Processing
follows STR input order, unmatched STR files are skipped, and extra VI files
are ignored, matching rOPTRAM. Use `output_csv` to persist the
returned table to a caller-selected CSV path.

### VI–STR compatibility notes

The remaining compatibility differences in this workflow are:

- Python returns `X`, `Y`, and `NDVI` plus pixel/source provenance columns;
  rOPTRAM's implementation returns lowercase `x`, `y`, and generic `VI`.
- Python requires identical VI/STR grids. rOPTRAM joins raster values by
  coordinates and can therefore create a partial intersection.
- Python always filters non-finite values, VI outside `[-1, 1]`, and
  non-positive STR. Its optional low-VI, high-STR, and SCL filtering behavior
  is unchanged.
- Like rOPTRAM, `max_tbl_size` defaults to one million rows, is divided across
  scenes, and randomly samples each oversized scene. Python's `max_rows` is an
  additional optional final sample.
- Python writes a caller-named CSV only when `output_csv` is supplied;
  rOPTRAM always writes `VI_STR_data.rds` to its output directory.
- Python returns an empty dataframe with a stable schema when matched scenes
  contain no valid pixels. rOPTRAM generally skips empty scenes and uses
  `NULL` for several empty-input cases.
- Like rOPTRAM intends, feature input labels pixels for feature plot coloring
  without changing the VI-STR population. Pixels outside features remain with
  a missing `Feature_ID`.

Apparent rOPTRAM implementation bugs in feature joining, vegetation-index
scaling, and exponential coefficient interpretation are not reproduced.

## Available API

- `get_cdse_token`
- `acquire_optram_inputs`
- `optram`
- `calculate_str`
- `optram_calculate_str`
- `optram_ndvi_str`
- `optram_options`
- `optram_wetdry_coefficients`
- `calculate_soil_moisture`
- `optram_calculate_soil_moisture`
- `plot_vi_str_cloud`

## Current status

This package is functional for coefficient generation, but still under active development toward fuller feature parity with rOPTRAM.

Planned additions include:
- a one-call `optram(...)` wrapper,
- broader documentation and tests.
