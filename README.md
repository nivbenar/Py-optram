# pyoptram

`pyoptram` is a Python implementation of the core OPTRAM workflow, developed
with behavioral parity with [rOPTRAM](https://github.com/ropensci/rOPTRAM) as
the primary goal.

The current package can acquire Sentinel-2 inputs from the Copernicus Data
Space Ecosystem (CDSE), construct a VI-STR pixel table, fit wet and dry
trapezoid edges, and calculate soil-moisture rasters through the lower-level
API. The high-level `optram()` wrapper follows the scope of
`rOPTRAM::optram()`: it stops after fitting and returns RMSE; it does not
calculate soil moisture.

## Installation

For an editable installation from this repository:

```bash
pip install -e .
```

GeoPandas is a runtime dependency because public acquisition and wrapper AOIs
are `geopandas.GeoDataFrame` objects.

## CDSE credentials

Store CDSE credentials once before running acquisition:

```python
from pyoptram import store_cdse_credentials

store_cdse_credentials(
    client_id="YOUR_CDSE_CLIENT_ID",
    client_secret="YOUR_CDSE_CLIENT_SECRET",
)
```

They are saved in the platform-specific user credential location and reused by
later Python or Jupyter sessions. A CSV-style file containing one
`clientid,secret` row can be imported instead:

```python
from pyoptram import store_cdse_credentials_from_file

store_cdse_credentials_from_file("path/to/cdse_credentials.csv")
```

If `OAUTH_CLIENTID` and `OAUTH_SECRET` are set, calling
`store_cdse_credentials()` without arguments imports and stores them. OAuth
tokens are requested as needed and are not persisted by the package.

## Quick start

The caller reads the GIS file and passes a GeoDataFrame. Py-optram does not
accept a filename, raw GeoJSON dictionary, or bounding-box tuple as the public
AOI input.

```python
import geopandas as gpd

from pyoptram import optram, optram_options

aoi = gpd.read_file("path/to/aoi.gpkg")

# Session options used by the child workflow functions.
optram_options("veg_index", "NDVI", show_opts=False)
optram_options("period", "full", show_opts=False)
optram_options("scm_mask", True, show_opts=False)

rmse = optram(
    aoi=aoi,
    from_date="2024-01-01",
    to_date="2024-03-31",
    s2_output_dir="data/optram/s2",
    data_output_dir="data/optram/results",
)
```

For the supported CDSE path, `optram()`:

1. acquires VI and STR imagery;
2. discovers the resulting files from their output directories;
3. constructs `VI_STR_data.parquet`;
4. fits the wet and dry edges;
5. prints and returns the wet/dry RMSE dataframe.

Matching `rOPTRAM::optram()`, it does **not** calculate soil moisture. Soil
moisture is available through the explicit lower-level API described below.

## Workflow

Acquisition accepts a non-empty Polygon or MultiPolygon GeoDataFrame with a
defined CRS. Multiple features are transformed to EPSG:4326 and geometrically
unioned for Catalog and Process requests. The original GeoDataFrame remains
unchanged and is forwarded to VI-STR feature labeling.

The CDSE workflow:

- consumes all Catalog pages by default;
- requires `to_date > from_date`;
- applies strict cloud filtering (`cloud < max_cloud`);
- uses case-sensitive tile substring matching;
- calculates spherical S2 area coverage, rounds it to three decimals, and
  retains coverage `>= area_cover`;
- optionally applies rOPTRAM's seasonal date filter;
- requests each selected acquisition's whole UTC day with
  `mosaickingOrder="mostRecent"`;
- defaults to 10-metre resolution and accepts 10, 20, or 60 metres.

BOA is downloaded by default and can be disabled with the `only_vi_str` option.
The current high-level wrapper supports the CDSE backend. rOPTRAM's openEO and
full remote-backend behavior are not yet implemented.

## Package options

`optram_options()` manages the currently implemented subset of rOPTRAM-style
session options. It is not a claim of complete rOPTRAM option-state parity.

| Option | Default |
|---|---:|
| `veg_index` | `"NDVI"` |
| `period` | `"full"` |
| `max_cloud` | `12` |
| `vi_step` | `0.005` |
| `trapezoid_method` | `"linear"` |
| `SWIR_band` | `11` |
| `max_tbl_size` | `1_000_000` |
| `rm.low.vi` | `False` |
| `rm.hi.str` | `False` |
| `plot_colors` | `"no"` |
| `feature_col` | `"ID"` |
| `edge_points` | `True` |
| `only_vi_str` | `False` |
| `tileid` | `None` |
| `scm_mask` | `True` |
| `overwrite` | `False` |
| `save_img_list` | `False` |
| `resolution` | `10` |
| `area_cover` | `99.0` |
| `porosity` | `0.4` |

```python
from pyoptram import optram_options

current = optram_options(show_opts=False)
optram_options("SWIR_band", 12, show_opts=False)
optram_options(reset=True, show_opts=False)
```

Where a public child function exposes an option-backed argument, an explicit
argument overrides the current session value.

## Advanced usage

The workflow stages can also be called individually:

```python
from pyoptram import (
    acquire_optram_inputs,
    optram_calculate_soil_moisture,
    optram_ndvi_str,
    optram_wetdry_coefficients,
)

acquired = acquire_optram_inputs(
    aoi=aoi,
    from_date="2024-01-01",
    to_date="2024-03-31",
    output_dir="data/optram/s2",
)

vi_str = optram_ndvi_str(
    ndvi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    output_parquet="data/optram/results/VI_STR_data.parquet",
    features=aoi,
)

rmse, coefficients, edge_points = optram_wetdry_coefficients(
    vi_str,
    output_dir="data/optram/results",
    return_outputs=True,
)

sm_paths = optram_calculate_soil_moisture(
    vi_paths=acquired["NDVI"],
    str_paths=acquired["STR"],
    coefficients=coefficients,
    output_dir="data/optram/soil_moisture",
)
```

Soil moisture defaults to porosity `0.4` and is not clipped, matching
rOPTRAM's default behavior. Pass `clip=True` explicitly to bound finite output
to `[0, porosity]`.

### VI-STR filename pairing

`optram_ndvi_str()` follows an STR-driven matching workflow. It processes STR
paths in supplied order, removes `STR_` from each STR basename, and searches VI
basenames using that identifier. For example,
`STR_2022-11-11_T36RXV.tif` matches
`NDVI_2022-11-11_T36RXV.tif`.

- An STR file with no VI match is skipped.
- An extra VI file with no STR match is ignored.
- More than one VI match for an STR raises `ValueError`.
- `max_tbl_size` is divided across the original number of STR inputs.
- Individual AOI features can supply `Feature_ID` labels when
  `plot_colors="features"`; they do not change the valid VI-STR population.

## Vegetation indices and evalscripts

CDSE acquisition supports these packaged Process API indices:

- NDVI
- SAVI
- MSAVI

The separate JavaScript files are packaged under `pyoptram/evalscripts/`:

- `BOA.js`
- `NDVI.js`, `NDVI_masked.js`
- `SAVI.js`, `SAVI_masked.js`
- `MSAVI.js`, `MSAVI_masked.js`
- `STR11.js`, `STR12.js`

The rOPTRAM option `scm_mask=True`—exposed as the acquisition argument
`scl_mask=True`—selects the masked VI evalscript. SCL masking occurs inside
that script and retains exactly classes `[2, 4, 5, 10]`. There is no separate
SCL raster download or local SCL-masking stage in `optram_ndvi_str()`. STR is
not directly SCL-masked; pixels with masked/non-finite VI values are removed by
the normal VI-STR validity filter.

The local `calculate_vi()` function additionally supports CI and BSCI, so its
complete set is NDVI, SAVI, MSAVI, CI, and BSCI. CI and BSCI are local
calculations only and cannot be requested through CDSE acquisition.

## Outputs

With the default acquisition options, the S2 output directory contains:

```text
s2/
├── <veg_index>/*.tif
├── STR/*.tif
└── BOA/*.tif
```

`BOA/` is omitted when `only_vi_str=True`. When `save_img_list=True`, the
post-filter Catalog is saved before downloads as `image_list.json`.

The high-level wrapper writes these data/fitting artifacts:

```text
results/
├── VI_STR_data.parquet
├── trapezoid_points.csv
├── wetdry_coefficients.csv
└── wetdry_rmse.csv
```

Manual fitting with `export_roptram=True` additionally writes
`coefficients_lin.csv` or `coefficients_pol.csv`. Manual soil-moisture
calculation writes `SM_*.tif` files to its requested output directory.

## rOPTRAM compatibility notes

Behavioral parity is the project priority, but full rOPTRAM feature parity is
not yet claimed. Current documented differences include:

- Python's optional Process API `width`/`height` mode is an additional
  compatibility path; the normal path uses rOPTRAM/CDSE-style resolution.
- `save_img_list=True` writes a GeoJSON `image_list.json`; rOPTRAM writes an
  RDS object.
- Area coverage uses `s2rst` rather than R `sf`/S2. Tiny floating-point
  differences can occur at artificial rounding boundaries; practical parity
  is defined after three-decimal rounding.
- Python's VI-STR table uses `X`, `Y`, and `NDVI` plus source/pixel provenance;
  rOPTRAM uses lowercase `x`, `y`, and generic `VI` columns.
- Python requires identical VI/STR grids. rOPTRAM joins values by coordinates
  and may therefore produce a partial intersection.
- Python filters non-finite values, VI outside `[-1, 1]`, and non-positive STR.
- `max_tbl_size` follows the rOPTRAM per-STR-input sampling approach; Python's
  `max_rows` is an additional optional final sample.
- Python writes a caller-selected Parquet table only when `output_parquet` is
  supplied; rOPTRAM writes `VI_STR_data.rds`. R can read the Python table with
  `arrow::read_parquet()`.
- Python returns a stable empty dataframe when matched inputs contain no valid
  pixels; rOPTRAM skips empty inputs and uses `NULL` in several such paths.
- Feature labeling implements rOPTRAM's intended feature-color behavior
  without reproducing its broken dataframe join.
- Local `calculate_vi()` applies the approved single scaling pass and requires
  only the bands used by the selected index, rather than reproducing
  rOPTRAM's apparent double-scaling and 12-band requirements.
- Native Python exponential coefficients use a consistent fitting/application
  equation. Export/import of rOPTRAM exponential coefficient files is not
  supported because the R export and application equations are inconsistent.

## Tests

Run the test suite with:

```bash
python -m pytest
```

The current suite collects 56 tests covering options, AOI handling, CDSE scene
selection and Process payloads, packaged evalscripts, VI-STR construction,
vegetation-index and STR formulas, trapezoid fitting, the wrapper, and
soil-moisture calculations.

## Current status

- The core Sentinel-2/CDSE acquisition-to-RMSE workflow is implemented.
- The public `optram()` wrapper is implemented and has run successfully
  end-to-end against real CDSE data.
- A Bushland validation produced RMSE wet `0.157336` and RMSE dry `0.024278`,
  compared with rOPTRAM's `0.1573351` and `0.02427703`. This demonstrates close
  agreement for that benchmark; it is not evidence of complete parity across
  every rOPTRAM feature and input path.
- Further numerical validation and remaining parity work are ongoing.
- openEO and full remote-backend parity are not yet implemented.
