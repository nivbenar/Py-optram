### Yuma OPTRAM Pipeline Example
# Demonstrates acquisition, VI-STR table assembly, trapezoid fitting, and
# plotting for a Yuma-area GeoDataFrame.

"""
Notebook-style pyOPTRAM test workflow saved as a Python script.

Run this after storing credentials once with store_cdse_credentials().
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pyoptram as op
from shapely.geometry import box


aoi = gpd.GeoDataFrame(
    geometry=[box(-114.75, 32.55, -114.45, 32.75)],
    crs="EPSG:4326",
)


### Download paired NDVI and STR rasters
results = op.acquire_optram_inputs(
    aoi=aoi,
    from_date="2025-12-01",
    to_date="2026-03-15",
    output_dir="outputs_yuma",
    veg_index="NDVI",
    swir_band=12,
    max_cloud=20,
    only_vi_str=True,
    limit=8,
    width=1024,
    height=1024,
)

print("NDVI files:", len(results["NDVI"]))
print("STR files:", len(results["STR"]))


### Build the VI-STR dataframe from the raster pairs
df = op.optram_ndvi_str(
    results["NDVI"],
    results["STR"],
    output_parquet="outputs_yuma/VI_STR_data.parquet",
)

print(df[["VI", "STR"]].describe())
print("Zero STR rows:", (df["STR"] == 0).sum())


### Fit the wet and dry trapezoid edges
rmse_df, coeffs_df, edges_df = op.optram_wetdry_coefficients(
    df,
    output_dir="outputs_yuma",
    method="linear",
    vi_step=0.05,
    rm_low_vi=True,
    return_outputs=True,
)

print(rmse_df)
print(coeffs_df)


### Plot the VI-STR cloud and fitted wet/dry edges
op.plot_vi_str_cloud(
    df,
    edges_df,
    edge_points=True,
    plot_colors="density",
    output_path="outputs_yuma/vi_str_trapezoid.png",
)

plt.show()
