### VI-STR Pixel-Table Assembly
# Pairs vegetation-index and STR rasters by filename, filters their pixels,
# and assembles spatial and file metadata in a dataframe.

import json
import re
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

from .options import _UNSET, _resolve_optram_option

_BASE_COLUMNS = [
    "X",
    "Y",
    "TimestampUTC",
    "Month",
    "Tile",
    "NDVI",
    "STR",
    "source_index",
    "row",
    "col",
    "ndvi_path",
    "str_path",
]

### Normalize one path or an iterable of paths to a non-empty list.
def _as_path_list(paths, name):
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError(f"{name} must contain at least one path")

    return path_list


### Read raster band 1 and convert NoData values to NaN.
def _read_band(path):
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = {
            "shape": array.shape,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
        }

    if profile["nodata"] is not None:
        array = np.where(array == profile["nodata"], np.nan, array)

    return array, profile


### Verify that two rasters describe the same pixel grid.
def _check_same_grid(reference_profile, other_profile, reference_path, other_path):
    for key, label in [("shape", "Shape"), ("transform", "Transform"), ("crs", "CRS")]:
        if reference_profile[key] != other_profile[key]:
            raise ValueError(f"{label} mismatch: {reference_path} and {other_path}")


### Parse a timestamp and Sentinel tile from a pyOPTRAM filename.
def _file_metadata(path):
    name = Path(path).stem
    timestamp_match = re.search(
        r"_(\d{4}-\d{2}-\d{2})(?:T(\d{2}-\d{2}-\d{2}(?:\.\d+)?))?",
        name,
    )
    tile_match = re.search(r"_T(\d{2}[A-Z]{3})(?:_|$)", name)

    timestamp = pd.NaT
    if timestamp_match:
        date_text = timestamp_match.group(1)
        time_text = (timestamp_match.group(2) or "00-00-00").replace("-", ":")
        timestamp = pd.to_datetime(f"{date_text} {time_text}", utc=True)

    month = timestamp.month if pd.notna(timestamp) else pd.NA
    tile = tile_match.group(1) if tile_match else pd.NA

    return timestamp, month, tile


### Feature filtering

### Load feature input as a list of GeoJSON Feature dictionaries.
def _load_features(features):
    if features is None:
        return None

    if isinstance(features, (str, Path)):
        path = Path(features)
        if not path.exists():
            raise FileNotFoundError(f"Features file not found: {path}")

        if path.suffix.lower() in (".geojson", ".json"):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            try:
                import geopandas as gpd
            except ImportError as exc:
                raise ImportError(
                    "Reading this features file type requires geopandas. "
                    "Install geopandas or pass a GeoJSON file/dict."
                ) from exc

            table = gpd.read_file(path)
            if table.crs is not None and table.crs.to_epsg() != 4326:
                table = table.to_crs(4326)
            data = json.loads(table.to_json())

    elif isinstance(features, dict):
        data = features
    else:
        raise TypeError(
            "features must be a GeoJSON dict/Feature/FeatureCollection or a "
            "path to a vector file"
        )

    if data.get("type") == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}

    if data.get("type") != "FeatureCollection" or "features" not in data:
        raise ValueError("features must be a GeoJSON Feature or FeatureCollection")

    if not data["features"]:
        raise ValueError("features contains no geometries")

    return data["features"]


### Burn feature IDs onto a raster grid for per-pixel membership lookup.
# NaN represents pixels outside all features.
def _rasterize_features(features, feature_id_col, transform, shape):
    from rasterio import features as rio_features

    shapes = []
    for feature in features:
        properties = feature.get("properties") or {}
        value = properties.get(feature_id_col)
        if value is None:
            continue
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"Feature property {feature_id_col!r} must be numeric")

        shapes.append((feature["geometry"], value))

    if not shapes:
        return np.full(shape, np.nan, dtype=np.float64)

    return rio_features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        all_touched=True,
        dtype="float64",
    )


### VI-STR table assembly

### Remove high STR outliers above Q3 + 1.5 * IQR.
def _remove_high_str(dataframe):
    q1 = dataframe["STR"].quantile(0.25)
    q3 = dataframe["STR"].quantile(0.75)
    cutoff = q3 + 1.5 * (q3 - q1)

    return dataframe[dataframe["STR"] < cutoff].copy()


### Assemble paired VI and STR raster pixels into a filtered dataframe.
def optram_ndvi_str(
    ndvi_paths,
    str_paths,
    output_csv=None,
    rm_low_vi=_UNSET,
    rm_hi_str=_UNSET,
    features=None,
    feature_id_col=_UNSET,
    plot_colors=_UNSET,
    max_tbl_size=_UNSET,
    max_rows=None,
    random_state=None,
):
    """Build a dataframe of paired NDVI and STR pixel values.

    Parameters
    ----------
    ndvi_paths, str_paths : path or list of paths
        VI and STR rasters matched from each STR basename, in STR input order.
        STR files without a matching VI are skipped.
    output_csv : path, optional
        If given, write the resulting dataframe to this CSV path.
    rm_low_vi : bool, optional
        Drop pixels with NDVI <= 0.005. Defaults to the ``rm.low.vi`` option,
        initially false.
    rm_hi_str : bool, optional
        Drop STR values at or above Q3 + 1.5 * IQR. Defaults to the
        ``rm.hi.str`` option, initially false.
    features : dict or path, optional
        A GeoJSON Feature/FeatureCollection (dict) or a path to a vector
        file. Features label pixels only when ``plot_colors`` is ``"feature"``
        or ``"features"``; they never change the valid VI-STR population.
        Pixels outside all features receive a missing ``Feature_ID``.
    feature_id_col : str, optional
        Property name in `features` used to populate Feature_ID. Defaults to
        rOPTRAM's ``"ID"`` option.
    plot_colors : str, optional
        Plot-color mode. Feature IDs are prepared only for ``"feature"`` or
        ``"features"``, matching rOPTRAM.
    max_tbl_size : int, optional
        Maximum table size, divided evenly across input scenes. Oversized
        scenes are randomly sampled to their share. Defaults to 1,000,000.
    max_rows : int, optional
        If the assembled (and filtered) table exceeds this many rows, it is
        randomly downsampled to this size.
    random_state : int, optional
        Seed used for per-file and max_rows downsampling.

    Raises
    ------
    ValueError
        If more than one VI matches an STR, grids differ, or a configured
        value is invalid.

    Returns
    -------
    pandas.DataFrame
        Valid paired pixels with coordinates, timestamp/month/tile metadata,
        VI and STR values, source/pixel provenance, and an optional
        ``Feature_ID`` column.

    Notes
    -----
    Finite VI values in [-1, 1] and positive STR values are retained before
    configured filters. ``max_tbl_size`` is divided evenly across STR files and
    oversized STR files are randomly sampled. Unlike rOPTRAM, output is written
    only when ``output_csv`` is supplied, and the format is CSV rather than
    RDS. Feature labeling implements rOPTRAM's intended behavior without its
    broken dataframe join.
    """
    rm_low_vi = _resolve_optram_option(
        "rm.low.vi", rm_low_vi, "rm_low_vi must be a boolean"
    )
    rm_hi_str = _resolve_optram_option(
        "rm.hi.str", rm_hi_str, "rm_hi_str must be a boolean"
    )
    feature_id_col = _resolve_optram_option(
        "feature_col", feature_id_col, "feature_id_col must be a string"
    )
    plot_colors = _resolve_optram_option(
        "plot_colors",
        plot_colors,
        "plot_colors must be a supported rOPTRAM plotting mode",
    )
    max_tbl_size = _resolve_optram_option(
        "max_tbl_size",
        max_tbl_size,
        "max_tbl_size must be numeric and at least 10000",
    )

    # Prepare input paths.
    ndvi_path_list = _as_path_list(ndvi_paths, "ndvi_paths")
    str_path_list = _as_path_list(str_paths, "str_paths")

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer")

    feature_list = None
    if plot_colors in {"feature", "features"} and features is not None:
        loaded_features = _load_features(features)
        if any(feature_id_col in (feature.get("properties") or {})
               for feature in loaded_features):
            feature_list = loaded_features
    frames = []
    rasterized_features_cache = {}
    scene_cap = int(max_tbl_size / len(str_path_list))
    rng = np.random.default_rng(random_state)

    for source_index, str_path in enumerate(str_path_list):
        unique_str = str_path.name.replace("STR_", "")
        vi_matches = [
            path for path in ndvi_path_list if re.search(unique_str, path.name)
        ]
        if not vi_matches:
            continue
        if len(vi_matches) > 1:
            raise ValueError(
                f"More than one VI file matches STR file {str_path.name!r}"
            )
        ndvi_path = vi_matches[0]
        # Read and validate one NDVI/STR raster pair.
        ndvi, ndvi_profile = _read_band(ndvi_path)
        str_array, str_profile = _read_band(str_path)
        _check_same_grid(ndvi_profile, str_profile, ndvi_path, str_path)

        # Keep only usable OPTRAM pixels.
        valid = (
            np.isfinite(ndvi)
            & np.isfinite(str_array)
            & (ndvi >= -1)
            & (ndvi <= 1)
            & (str_array > 0)
        )

        if rm_low_vi:
            valid &= ndvi > 0.005

        feature_burn = None
        if feature_list is not None:
            cache_key = (ndvi_profile["transform"], ndvi_profile["shape"])
            feature_burn = rasterized_features_cache.get(cache_key)
            if feature_burn is None:
                feature_burn = _rasterize_features(
                    feature_list,
                    feature_id_col,
                    ndvi_profile["transform"],
                    ndvi_profile["shape"],
                )
                rasterized_features_cache[cache_key] = feature_burn

        rows, cols = np.where(valid)
        if len(rows) == 0:
            continue

        if len(rows) > scene_cap:
            selected = rng.choice(len(rows), size=scene_cap, replace=False)
            rows = rows[selected]
            cols = cols[selected]

        # Convert raster pixels to dataframe rows.
        xs, ys = xy(ndvi_profile["transform"], rows, cols)
        timestamp, month, tile = _file_metadata(ndvi_path)

        row_data = {
            "X": xs,
            "Y": ys,
            "TimestampUTC": timestamp,
            "Month": month,
            "Tile": tile,
            "NDVI": ndvi[rows, cols],
            "STR": str_array[rows, cols],
            "source_index": source_index,
            "row": rows,
            "col": cols,
            "ndvi_path": str(ndvi_path),
            "str_path": str(str_path),
        }

        if feature_burn is not None:
            row_data["Feature_ID"] = feature_burn[rows, cols]

        frames.append(pd.DataFrame(row_data))

    columns = list(_BASE_COLUMNS)
    if feature_list is not None:
        columns.append("Feature_ID")

    if not frames:
        return pd.DataFrame(columns=columns)

    dataframe = pd.concat(frames, ignore_index=True)[columns]

    if rm_hi_str:
        dataframe = _remove_high_str(dataframe)

    if max_rows is not None and len(dataframe) > max_rows:
        dataframe = dataframe.sample(n=max_rows, random_state=random_state)
        dataframe = dataframe.reset_index(drop=True)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_csv, index=False)

    return dataframe
