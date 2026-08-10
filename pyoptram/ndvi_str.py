import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

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


def _as_path_list(paths, name):
    # Accept one path or a list of paths.
    if isinstance(paths, (str, Path)):
        return [Path(paths)]

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError(f"{name} must contain at least one path")

    return path_list


def _as_optional_path_list(paths, name, expected_len):
    # Same as _as_path_list, but allows None and enforces a matching length.
    if paths is None:
        return None

    path_list = _as_path_list(paths, name)
    if len(path_list) != expected_len:
        raise ValueError(f"{name} must have the same length as ndvi_paths")

    return path_list


def _read_band(path):
    # Read raster band 1 and convert NoData to NaN.
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


def _read_scl(path):
    # Read the Scene Classification Layer as integer class codes.
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = {
            "shape": array.shape,
            "transform": src.transform,
            "crs": src.crs,
        }

    return array, profile


def _check_same_grid(reference_profile, other_profile, reference_path, other_path):
    # Two rasters must describe exactly the same pixel grid.
    for key, label in [("shape", "Shape"), ("transform", "Transform"), ("crs", "CRS")]:
        if reference_profile[key] != other_profile[key]:
            raise ValueError(f"{label} mismatch: {reference_path} and {other_path}")


def _file_metadata(path):
    # Pull timestamp and Sentinel tile from the pyoptram filename (fallback
    # used when no scene_metadata record is available for this file).
    name = Path(path).stem
    timestamp_match = re.search(
        r"_(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2}(?:\.\d+)?)",
        name,
    )
    tile_match = re.search(r"_T(\d{2}[A-Z]{3})_", name)

    timestamp = pd.NaT
    if timestamp_match:
        date_text = timestamp_match.group(1)
        time_text = timestamp_match.group(2).replace("-", ":")
        timestamp = pd.to_datetime(f"{date_text} {time_text}", utc=True)

    month = timestamp.month if pd.notna(timestamp) else pd.NA
    tile = tile_match.group(1) if tile_match else pd.NA

    return timestamp, month, tile


def _build_scene_lookup(scene_metadata):
    # Index scene records (as returned by acquire_optram_inputs) by every
    # raster path they reference, so metadata can be recovered without
    # re-parsing filenames.
    if not scene_metadata:
        return {}

    lookup = {}
    for record in scene_metadata:
        for key in ("NDVI", "STR", "BOA", "SCL"):
            path = record.get(key)
            if path:
                lookup[str(Path(path))] = record

    return lookup


def _metadata_from_scene_record(record):
    timestamp = pd.to_datetime(record.get("datetime"), utc=True, errors="coerce")
    month = timestamp.month if pd.notna(timestamp) else pd.NA
    tile = record.get("tile")
    tile = tile if tile else pd.NA
    return timestamp, month, tile


def _scene_metadata_for(ndvi_path, str_path, scene_lookup):
    # Prefer an exact scene_metadata match; fall back to filename parsing.
    record = scene_lookup.get(str(ndvi_path)) or scene_lookup.get(str(str_path))
    if record is not None:
        return _metadata_from_scene_record(record)
    return _file_metadata(ndvi_path)


def _load_features(features):
    # Normalize a GeoJSON dict, Feature, FeatureCollection, or vector file
    # path into a plain list of GeoJSON Feature dicts.
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


def _rasterize_features(features, feature_id_col, transform, shape):
    # Burn each feature's ID onto the raster grid so per-pixel membership is
    # a single array lookup. Returns an int32 array where 0 means "outside
    # every feature".
    from rasterio import features as rio_features

    shapes = []
    for index, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        value = properties.get(feature_id_col) if feature_id_col else None

        if value is None:
            value = index
        else:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = index

        if value == 0:
            # 0 is reserved to mean "no feature"; shift real IDs off it.
            value = index

        shapes.append((feature["geometry"], value))

    return rio_features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )


def _remove_high_str(dataframe):
    # Remove high STR outliers with Q3 + 1.5 * IQR.
    q1 = dataframe["STR"].quantile(0.25)
    q3 = dataframe["STR"].quantile(0.75)
    cutoff = q3 + 1.5 * (q3 - q1)

    return dataframe[dataframe["STR"] < cutoff].copy()


def optram_ndvi_str(
    ndvi_paths,
    str_paths,
    output_csv=None,
    rm_low_vi=False,
    rm_hi_str=False,
    scl_paths=None,
    scl_keep=None,
    features=None,
    feature_id_col=None,
    max_tbl_size=None,
    max_rows=None,
    scene_metadata=None,
    random_state=None,
):
    """Build a dataframe of paired NDVI and STR pixel values.

    Parameters
    ----------
    ndvi_paths, str_paths : path or list of paths
        Matching NDVI and STR rasters, one pair per scene.
    output_csv : path, optional
        If given, write the resulting dataframe to this CSV path.
    rm_low_vi : bool
        Drop pixels with NDVI <= 0.005 (thin/absent vegetation).
    rm_hi_str : bool
        Drop STR outliers above Q3 + 1.5 * IQR.
    scl_paths : path or list of paths, optional
        Sentinel-2 Scene Classification Layer rasters, one per scene, used
        to mask out clouds/shadows/snow. Must be on the same grid as the
        matching NDVI/STR pair.
    scl_keep : iterable of int, optional
        SCL class codes to keep. Defaults to {4, 5, 6, 7} (vegetation, bare
        soils, water, unclassified). Only used when scl_paths is given.
    features : dict or path, optional
        A GeoJSON Feature/FeatureCollection (dict) or a path to a vector
        file. When given, only pixels that fall inside a feature are kept,
        and a Feature_ID column is added.
    feature_id_col : str, optional
        Property name in `features` used to populate Feature_ID. Falls back
        to a 1-based feature index when omitted or missing on a feature.
    max_tbl_size : int, optional
        Hard cap enforced while assembling the table: once this many rows
        have been collected, remaining scenes are skipped. Bounds memory
        use for very large jobs.
    max_rows : int, optional
        If the assembled (and filtered) table exceeds this many rows, it is
        randomly downsampled to this size.
    scene_metadata : list of dict, optional
        The "scenes" records returned by acquire_optram_inputs. When given,
        TimestampUTC/Month/Tile are looked up from these records instead of
        being parsed from filenames, which is more robust.
    random_state : int, optional
        Seed used for max_rows downsampling.
    """
    # Prepare input paths.
    ndvi_path_list = _as_path_list(ndvi_paths, "ndvi_paths")
    str_path_list = _as_path_list(str_paths, "str_paths")

    if len(ndvi_path_list) != len(str_path_list):
        raise ValueError("ndvi_paths and str_paths must have the same length")

    scl_path_list = _as_optional_path_list(scl_paths, "scl_paths", len(ndvi_path_list))

    if max_tbl_size is not None and max_tbl_size < 1:
        raise ValueError("max_tbl_size must be a positive integer")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer")

    scl_keep_set = DEFAULT_SCL_KEEP if scl_keep is None else frozenset(int(v) for v in scl_keep)
    feature_list = _load_features(features)
    scene_lookup = _build_scene_lookup(scene_metadata)

    frames = []
    total_rows = 0
    rasterized_features_cache = {}

    for source_index, (ndvi_path, str_path) in enumerate(zip(ndvi_path_list, str_path_list)):
        if max_tbl_size is not None and total_rows >= max_tbl_size:
            break

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
            scl_path = scl_path_list[source_index]
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
            valid &= feature_burn > 0

        rows, cols = np.where(valid)
        if len(rows) == 0:
            continue

        if max_tbl_size is not None and total_rows + len(rows) > max_tbl_size:
            keep_n = max_tbl_size - total_rows
            rows = rows[:keep_n]
            cols = cols[:keep_n]

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
        total_rows += len(rows)

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
