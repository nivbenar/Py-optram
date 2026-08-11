### VI-STR Pixel-Table Assembly
# Pairs vegetation-index and STR rasters by scene, filters their pixels, and
# assembles spatial and scene metadata in a dataframe.

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

# rOPTRAM default: keep vegetation, bare soils, water, and unclassified pixels.
DEFAULT_SCL_KEEP = frozenset({4, 5, 6, 7})


### Normalize one path or an iterable of paths to a non-empty list.
def _as_path_list(paths, name):
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError(f"{name} must contain at least one path")

    return path_list


### Normalize optional paths while preserving None.
def _as_optional_path_list(paths, name):
    if paths is None:
        return None

    return _as_path_list(paths, name)


### Derive the filename portion shared by products from one scene.
def _scene_key(path):
    """Return the filename portion shared by products from one scene."""
    stem = Path(path).stem
    _, separator, key = stem.partition("_")
    if not separator or not key:
        raise ValueError(
            f"Cannot derive a scene key from filename {Path(path).name!r}; "
            "expected a product prefix followed by '_'"
        )
    return key


### Index product paths by scene key and reject duplicate keys.
def _paths_by_scene(paths, name):
    """Index paths by scene key, rejecting ambiguous product filenames."""
    indexed = {}
    for path in paths:
        key = _scene_key(path)
        if key in indexed:
            raise ValueError(
                f"Cannot match {name} uniquely for scene {key!r}: "
                f"{indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


### Pair VI, STR, and optional SCL paths while preserving STR order.
def _pair_scene_paths(ndvi_paths, str_paths, scl_paths=None):
    """Pair required scene products by filename, preserving STR order."""
    ndvi_by_scene = _paths_by_scene(ndvi_paths, "VI files")
    str_by_scene = _paths_by_scene(str_paths, "STR files")

    ndvi_keys = set(ndvi_by_scene)
    str_keys = set(str_by_scene)
    if ndvi_keys != str_keys:
        missing_vi = sorted(str_keys - ndvi_keys)
        missing_str = sorted(ndvi_keys - str_keys)
        details = []
        if missing_vi:
            details.append(f"missing VI for scenes: {missing_vi}")
        if missing_str:
            details.append(f"missing STR for scenes: {missing_str}")
        raise ValueError("Cannot match VI and STR files uniquely; " + "; ".join(details))

    scl_by_scene = None
    if scl_paths is not None:
        scl_by_scene = _paths_by_scene(scl_paths, "SCL files")
        scl_keys = set(scl_by_scene)
        if scl_keys != str_keys:
            missing_scl = sorted(str_keys - scl_keys)
            extra_scl = sorted(scl_keys - str_keys)
            details = []
            if missing_scl:
                details.append(f"missing SCL for scenes: {missing_scl}")
            if extra_scl:
                details.append(f"SCL has no VI/STR scene: {extra_scl}")
            raise ValueError("Cannot match SCL files uniquely; " + "; ".join(details))

    pairs = []
    for str_path in str_paths:
        key = _scene_key(str_path)
        scl_path = scl_by_scene[key] if scl_by_scene is not None else None
        pairs.append((ndvi_by_scene[key], str_path, scl_path))
    return pairs


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


### Read a Scene Classification Layer as integer class codes.
def _read_scl(path):
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = {
            "shape": array.shape,
            "transform": src.transform,
            "crs": src.crs,
        }

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


### Index acquisition scene records by every raster path they reference.
def _build_scene_lookup(scene_metadata):
    if not scene_metadata:
        return {}

    lookup = {}
    for record in scene_metadata:
        for key in ("NDVI", "STR", "BOA", "SCL"):
            path = record.get(key)
            if path:
                lookup[str(Path(path))] = record

    return lookup


### Extract timestamp, month, and tile values from a scene record.
def _metadata_from_scene_record(record):
    timestamp = pd.to_datetime(record.get("datetime"), utc=True, errors="coerce")
    month = timestamp.month if pd.notna(timestamp) else pd.NA
    tile = record.get("tile")
    tile = tile if tile else pd.NA
    return timestamp, month, tile


### Resolve scene metadata from acquisition records or the VI filename.
def _scene_metadata_for(ndvi_path, str_path, scene_lookup):
    record = scene_lookup.get(str(ndvi_path)) or scene_lookup.get(str(str_path))
    if record is not None:
        return _metadata_from_scene_record(record)
    return _file_metadata(ndvi_path)


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
    scl_paths=None,
    scl_keep=None,
    features=None,
    feature_id_col=_UNSET,
    plot_colors=_UNSET,
    max_tbl_size=_UNSET,
    max_rows=None,
    scene_metadata=None,
    random_state=None,
):
    """Build a dataframe of paired NDVI and STR pixel values.

    Parameters
    ----------
    ndvi_paths, str_paths : path or list of paths
        VI and STR rasters paired by the filename portion after the product
        prefix. Every scene must have exactly one VI and one STR file.
    output_csv : path, optional
        If given, write the resulting dataframe to this CSV path.
    rm_low_vi : bool, optional
        Drop pixels with NDVI <= 0.005. Defaults to the ``rm.low.vi`` option,
        initially false.
    rm_hi_str : bool, optional
        Drop STR values at or above Q3 + 1.5 * IQR. Defaults to the
        ``rm.hi.str`` option, initially false.
    scl_paths : path or list of paths, optional
        Sentinel-2 Scene Classification Layer rasters, one per scene, used
        to mask out clouds/shadows/snow. Must be on the same grid as the
        matching NDVI/STR pair.
    scl_keep : iterable of int, optional
        SCL class codes to keep. Defaults to {4, 5, 6, 7} (vegetation, bare
        soils, water, unclassified). Only used when scl_paths is given.
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
    scene_metadata : list of dict, optional
        The "scenes" records returned by acquire_optram_inputs. When given,
        TimestampUTC/Month/Tile are looked up from these records instead of
        being parsed from filenames, which is more robust.
    random_state : int, optional
        Seed used for per-scene and max_rows downsampling.

    Raises
    ------
    ValueError
        If required VI, STR, or supplied SCL files cannot be matched uniquely
        by scene filename, grids differ, or a configured value is invalid.

    Returns
    -------
    pandas.DataFrame
        Valid paired pixels with coordinates, timestamp/month/tile metadata,
        VI and STR values, source/pixel provenance, and optional ``SCL`` and
        ``Feature_ID`` columns.

    Notes
    -----
    Finite VI values in [-1, 1] and positive STR values are retained before
    configured filters. ``max_tbl_size`` is divided evenly across scenes and
    oversized scenes are randomly sampled. Unlike rOPTRAM, output is written
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

    scl_path_list = _as_optional_path_list(scl_paths, "scl_paths")
    scene_pairs = _pair_scene_paths(ndvi_path_list, str_path_list, scl_path_list)

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer")

    scl_keep_set = DEFAULT_SCL_KEEP if scl_keep is None else frozenset(int(v) for v in scl_keep)
    feature_list = None
    if plot_colors in {"feature", "features"} and features is not None:
        loaded_features = _load_features(features)
        if any(feature_id_col in (feature.get("properties") or {})
               for feature in loaded_features):
            feature_list = loaded_features
    scene_lookup = _build_scene_lookup(scene_metadata)

    frames = []
    rasterized_features_cache = {}
    scene_cap = int(max_tbl_size / len(scene_pairs))
    rng = np.random.default_rng(random_state)

    for source_index, (ndvi_path, str_path, scl_path) in enumerate(scene_pairs):
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

        scl_array = None
        if scl_path_list is not None:
            scl_array, scl_profile = _read_scl(scl_path)
            _check_same_grid(ndvi_profile, scl_profile, ndvi_path, scl_path)
            valid &= np.isin(scl_array, list(scl_keep_set))

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
        timestamp, month, tile = _scene_metadata_for(ndvi_path, str_path, scene_lookup)

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

        if scl_array is not None:
            row_data["SCL"] = scl_array[rows, cols]
        if feature_burn is not None:
            row_data["Feature_ID"] = feature_burn[rows, cols]

        frames.append(pd.DataFrame(row_data))

    columns = list(_BASE_COLUMNS)
    if scl_path_list is not None:
        columns.append("SCL")
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
