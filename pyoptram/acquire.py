### Copernicus Data Space Acquisition
# Stores CDSE credentials, searches the Sentinel-2 L2A catalog, and downloads
# vegetation-index, STR, BOA, and optional SCL rasters for OPTRAM workflows.

from datetime import datetime
from pathlib import Path
import csv
import json
import os
import platform
import warnings

import requests
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .options import _UNSET, _resolve_optram_option, get_optram_option


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEFAULT_MAX_SIZE = 2500


### Credential storage and retrieval

### Return the platform-specific CDSE credentials-file path.
def _cdse_credentials_file():
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return None
        credentials_dir = Path(base) / "CDSE"
    elif system == "Linux":
        credentials_dir = Path.home() / ".CDSE"
    elif system == "Darwin":
        credentials_dir = Path.home() / "Library" / "Preferences" / ".CDSE"
    else:
        return None
    return credentials_dir / "cdse_credentials.json"


### Store CDSE OAuth credentials in the shared rOPTRAM-compatible file.
def store_cdse_credentials(client_id=None, client_secret=None):
    """Store CDSE OAuth credentials in rOPTRAM's platform-specific file."""
    credentials_file = _cdse_credentials_file()
    if credentials_file is None:
        warnings.warn(
            "Platform is not identified. No credentials are saved",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if client_id is None or client_secret is None:
        client_id = os.environ.get("OAUTH_CLIENTID", "")
        client_secret = os.environ.get("OAUTH_SECRET", "")

    if not client_id or not client_secret:
        return None

    credentials_file.parent.mkdir(parents=True, exist_ok=True)
    credentials = [{"clientid": client_id, "secret": client_secret}]
    credentials_file.write_text(json.dumps(credentials), encoding="utf-8")
    return None


### Store CDSE OAuth credentials from a one-record CSV file.
def store_cdse_credentials_from_file(path):
    """Store CDSE OAuth credentials from a clientid,secret CSV file."""
    with Path(path).open(newline="", encoding="utf-8-sig") as credentials_file:
        reader = csv.DictReader(credentials_file)
        if reader.fieldnames is None or not {"clientid", "secret"}.issubset(reader.fieldnames):
            raise ValueError("Credentials file must contain clientid and secret headers")
        records = [row for row in reader if any(value and value.strip() for value in row.values())]

    if len(records) != 1:
        raise ValueError("Credentials file must contain exactly one credential record")

    client_id = records[0].get("clientid", "").strip()
    client_secret = records[0].get("secret", "").strip()
    if not client_id or not client_secret:
        raise ValueError("Credentials file must contain non-empty clientid and secret values")

    store_cdse_credentials(client_id, client_secret)
    return None


### Retrieve CDSE OAuth credentials from platform-specific storage.
def retrieve_cdse_credentials():
    """Retrieve CDSE OAuth credentials stored by Py-optram or rOPTRAM."""
    credentials_file = _cdse_credentials_file()
    if credentials_file is None:
        warnings.warn(
            "Platform is not identified. No credentials are available",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not credentials_file.exists():
        warnings.warn(
            f"No credentials file found at {credentials_file}. "
            "Credentials are not available.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    credentials = json.loads(credentials_file.read_text(encoding="utf-8"))
    if not isinstance(credentials, list) or not credentials:
        return None
    stored = credentials[0]
    if not isinstance(stored, dict):
        return None
    client_id = stored.get("clientid")
    client_secret = stored.get("secret")
    if not client_id or not client_secret:
        return None
    return {"clientid": client_id, "secret": client_secret}


### CDSE request helpers

### Raise an HTTP error that includes the server response body.
def _raise_for_status(response, context):
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = (
            f"{context} failed with HTTP {response.status_code}: "
            f"{response.text}"
        )
        raise requests.HTTPError(message, response=response) from exc


### Request a CDSE access token with client credentials.
def get_cdse_token(client_id, client_secret):
    response = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"CDSE token request failed with HTTP {response.status_code}",
            response=response,
        ) from exc
    return response.json()["access_token"]


### Input validation and preparation

### Validate and return a date string formatted as YYYY-MM-DD.
def validate_date(date_text, name):
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be formatted as YYYY-MM-DD") from exc

    return date_text


### Read and normalize the geometry from a vector file.
def _geometry_from_vector_file(path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Reading this AOI file type requires geopandas. Install geopandas "
            "or pass a GeoJSON file, GeoJSON dictionary, or bbox tuple."
        ) from exc

    data = gpd.read_file(path)

    if data.empty:
        raise ValueError(f"AOI file has no features: {path}")

    if data.crs is not None and data.crs.to_epsg() != 4326:
        data = data.to_crs(4326)

    geometry = data.geometry.unary_union
    return json.loads(gpd.GeoSeries([geometry], crs="EPSG:4326").to_json())[
        "features"
    ][0]["geometry"]


### Union every polygon in a GeoJSON FeatureCollection.
def _union_feature_collection(data):
    """Union all Polygon/MultiPolygon features into one GeoJSON geometry.

    Returns one Polygon or MultiPolygon used for acquisition AOI parity with
    rOPTRAM.
    """
    features = data.get("features")
    if not features:
        raise ValueError("AOI FeatureCollection must contain at least one feature")

    geometries = []
    for feature in features:
        if feature.get("type") != "Feature" or not feature.get("geometry"):
            raise ValueError("AOI FeatureCollection contains an invalid feature")
        geometry = shape(feature["geometry"])
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("AOI features must be Polygon or MultiPolygon geometries")
        geometries.append(geometry)

    return mapping(unary_union(geometries))


### Convert a GeoJSON object, vector file, or bbox to GeoJSON geometry.
def load_aoi(aoi):
    """Normalize an AOI, geometrically unioning GeoJSON FeatureCollections."""
    if isinstance(aoi, dict):
        if "type" not in aoi:
            raise ValueError("AOI dictionary must contain a GeoJSON 'type'.")
        if aoi["type"] == "Feature":
            return aoi["geometry"]
        if aoi["type"] == "FeatureCollection":
            return _union_feature_collection(aoi)
        return aoi

    if isinstance(aoi, (str, Path)):
        path = Path(aoi)
        if not path.exists():
            raise FileNotFoundError(f"AOI file not found: {path}")

        if path.suffix.lower() in (".geojson", ".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data["type"] == "FeatureCollection":
                return _union_feature_collection(data)
            if data["type"] == "Feature":
                return data["geometry"]
            return data

        return _geometry_from_vector_file(path)

    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        minx, miny, maxx, maxy = aoi
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]],
        }

    raise ValueError(
        "AOI must be a GeoJSON dict, a vector file path, or a bbox tuple "
        "(minx, miny, maxx, maxy)."
    )


### Create output folders for VI, STR, and optional BOA and SCL rasters.
def prepare_output_folders(output_dir, veg_index=_UNSET, only_vi_str=_UNSET,
                           download_scl=False):
    if veg_index is _UNSET:
        veg_index = get_optram_option("veg_index")
    if only_vi_str is _UNSET:
        only_vi_str = get_optram_option("only_vi_str")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = {"vi": output_dir / veg_index, "str": output_dir / "STR"}

    if not only_vi_str:
        folders["boa"] = output_dir / "BOA"

    if download_scl:
        folders["scl"] = output_dir / "SCL"

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


### Sentinel Hub scripts and catalog operations

### Return a Sentinel Hub evalscript for a supported VI, STR, BOA, or SCL.
def load_evalscript(script_name, swir_band=_UNSET, scl_mask=False, scl_keep=None):
    """Return a Sentinel Hub evalscript used by the Process API."""
    if swir_band is _UNSET:
        swir_band = get_optram_option("SWIR_band")
    vi_formulas = {
        "NDVI": "(sample.B08 - sample.B04) / (sample.B08 + sample.B04)",
        "SAVI": (
            "1.5 * (sample.B08 - sample.B04) / "
            "(sample.B08 + sample.B04 + 0.5)"
        ),
        "MSAVI": (
            "(2 * sample.B08 + 1 - "
            "Math.sqrt(Math.pow(2 * sample.B08 + 1, 2) - "
            "8 * (sample.B08 - sample.B04))) / 2"
        ),
    }

    if script_name in vi_formulas:
        variable_name = script_name.lower()
        formula = vi_formulas[script_name]
        if scl_mask:
            keep = [2, 4, 5, 10] if scl_keep is None else sorted(set(scl_keep))
            keep_js = ", ".join(str(int(value)) for value in keep)
            return f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: ["B04", "B08", "SCL"] }}],
        output: {{ bands: 1, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    let {variable_name} = {formula};
    if ([{keep_js}].includes(sample.SCL)) {{
        return [{variable_name}];
    }}
    return [NaN];
}}
""".strip()

        return f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: ["B04", "B08"] }}],
        output: {{ bands: 1, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    let {variable_name} = {formula};
    return [{variable_name}];
}}
""".strip()

    if script_name == "STR":
        band = f"B{swir_band}"
        return f"""
//VERSION=3
// Calculate SWIR Transformed Reflectance from Sentinel-2 B11 or B12.
function setup() {{
    return {{
        input: [{{ bands: ["{band}"], units: "DN" }}],
        output: {{ bands: 1, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    let value = sample.{band};
    if (value !== 0) {{
        let v = value / 10000.0;
        let str_value = ((1 - v) ** 2) / (2 * v);
        return [str_value];
    }}
    return [0];
}}
""".strip()

    if script_name == "BOA":
        return """
//VERSION=3
// Download Sentinel-2 bottom-of-atmosphere bands as one multiband raster.
function setup() {
    return {
        input: [{
            bands: ["B01","B02","B03","B04","B05","B06",
                    "B07","B08","B8A","B09","B11","B12"],
            units: "DN"
        }],
        output: { bands: 12, sampleType: "UINT16" }
    };
}
function evaluatePixel(sample) {
    return [
        sample.B01, sample.B02, sample.B03, sample.B04,
        sample.B05, sample.B06, sample.B07, sample.B08,
        sample.B8A, sample.B09, sample.B11, sample.B12
    ];
}
""".strip()

    if script_name == "SCL":
        return """
//VERSION=3
// Download the Sentinel-2 Scene Classification Layer.
function setup() {
    return {
        input: [{ bands: ["SCL"] }],
        output: { bands: 1, sampleType: "UINT8" }
    };
}
function evaluatePixel(sample) {
    return [sample.SCL];
}
""".strip()

    raise ValueError(f"Unknown script_name: {script_name}")


### Extract the Sentinel-2 MGRS tile from a catalog scene.
def _scene_tile(scene):
    properties = scene.get("properties", {})

    for key in ("s2:mgrs_tile", "grid:code"):
        value = properties.get(key)
        if value:
            return str(value).replace("MGRS-", "").upper()

    scene_id = scene.get("id", "")
    parts = scene_id.split("_")
    for part in parts:
        if part.startswith("T") and len(part) == 6:
            return part[1:].upper()

    return None


### Keep only scenes from the requested MGRS tile.
def _filter_scenes_by_tile(scenes, tile):
    if tile is None:
        return scenes

    wanted = str(tile).upper().removeprefix("T")
    return [scene for scene in scenes if _scene_tile(scene) == wanted]


### Search the Sentinel-2 L2A catalog for matching scenes.
def search_catalog(
    aoi_geometry,
    from_date,
    to_date,
    token,
    max_cloud=_UNSET,
    limit=20,
    tile=None,
):
    max_cloud = _resolve_optram_option(
        "max_cloud",
        max_cloud,
        "max_cloud must be numeric and between 0 and 100",
    )
    headers = {"Authorization": f"Bearer {token}"}
    request_limit = max(limit, 100) if tile is not None else limit

    payload = {
        "intersects": aoi_geometry,
        "datetime": f"{from_date}T00:00:00Z/{to_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": request_limit,
        "filter": f"eo:cloud_cover < {max_cloud}",
    }

    response = requests.post(CATALOG_URL, headers=headers, json=payload, timeout=60)
    _raise_for_status(response, "Sentinel-2 catalog search")
    scenes = response.json().get("features", [])
    return _filter_scenes_by_tile(scenes, tile)[:limit]


### Download one GeoTIFF product through the Sentinel Hub Process API.
def download_index(
    aoi_geometry,
    scene_datetime,
    script_name,
    output_path,
    token,
    swir_band=_UNSET,
    width=DEFAULT_MAX_SIZE,
    height=DEFAULT_MAX_SIZE,
    overwrite=_UNSET,
    scl_mask=False,
    scl_keep=None,
):
    if swir_band is _UNSET:
        swir_band = get_optram_option("SWIR_band")
    if overwrite is _UNSET:
        overwrite = get_optram_option("overwrite")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }

    payload = {
        "input": {
            "bounds": {"geometry": aoi_geometry},
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": scene_datetime,
                            "to": scene_datetime,
                        }
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": load_evalscript(
            script_name,
            swir_band=swir_band,
            scl_mask=scl_mask,
            scl_keep=scl_keep,
        ),
    }

    response = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=180)
    _raise_for_status(response, f"{script_name} download")

    output_path.write_bytes(response.content)
    return str(output_path)


### Build a metadata record for one downloaded scene.
def _scene_record(scene, veg_index, vi_path, str_path, boa_path=None, scl_path=None):
    properties = scene.get("properties", {})
    record = {
        "id": scene.get("id"),
        "datetime": properties.get("datetime"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "tile": _scene_tile(scene),
        "STR": str_path,
        "BOA": boa_path,
        "SCL": scl_path,
    }
    record[veg_index] = vi_path
    return record


### Acquire paired vegetation-index and STR rasters for an OPTRAM workflow.
def acquire_optram_inputs(
    aoi,
    from_date,
    to_date,
    output_dir,
    client_id=None,
    client_secret=None,
    save_creds=True,
    veg_index=_UNSET,
    swir_band=_UNSET,
    max_cloud=_UNSET,
    only_vi_str=_UNSET,
    tile=_UNSET,
    limit=20,
    width=DEFAULT_MAX_SIZE,
    height=DEFAULT_MAX_SIZE,
    overwrite=_UNSET,
    scl_mask=_UNSET,
    download_scl=False,
    scl_keep=None,
):
    veg_index = _resolve_optram_option(
        "veg_index",
        veg_index,
        "CDSE acquisition supports NDVI, SAVI, or MSAVI",
    )
    swir_band = _resolve_optram_option(
        "SWIR_band", swir_band, "swir_band must be 11 or 12"
    )
    max_cloud = _resolve_optram_option(
        "max_cloud",
        max_cloud,
        "max_cloud must be numeric and between 0 and 100",
    )
    only_vi_str = _resolve_optram_option(
        "only_vi_str", only_vi_str, "only_vi_str must be a boolean"
    )
    tile = _resolve_optram_option(
        "tileid", tile, "tile must be None or a five-character MGRS tile ID"
    )
    overwrite = _resolve_optram_option(
        "overwrite", overwrite, "overwrite must be a boolean"
    )
    scl_mask = _resolve_optram_option(
        "scm_mask", scl_mask, "scl_mask must be a boolean"
    )

    if veg_index not in ("NDVI", "SAVI", "MSAVI"):
        raise ValueError("CDSE acquisition supports NDVI, SAVI, or MSAVI")

    if width > DEFAULT_MAX_SIZE or height > DEFAULT_MAX_SIZE:
        raise ValueError("width and height cannot exceed 2500 pixels")

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive integers")

    from_date = validate_date(from_date, "from_date")
    to_date = validate_date(to_date, "to_date")

    if from_date >= to_date:
        raise ValueError("to_date must be later than from_date")

    aoi_geometry = load_aoi(aoi)

    explicit_credentials = client_id is not None and client_secret is not None
    if not explicit_credentials:
        stored_credentials = retrieve_cdse_credentials()
        if stored_credentials is None:
            raise ValueError(
                "No CDSE credentials are available. Supply client_id and "
                "client_secret, or call store_cdse_credentials() first."
            )
        client_id = stored_credentials["clientid"]
        client_secret = stored_credentials["secret"]

    token = get_cdse_token(client_id, client_secret)
    if explicit_credentials and save_creds:
        store_cdse_credentials(client_id, client_secret)
    folders = prepare_output_folders(
        output_dir,
        veg_index=veg_index,
        only_vi_str=only_vi_str,
        download_scl=download_scl,
    )

    scenes = search_catalog(
        aoi_geometry=aoi_geometry,
        from_date=from_date,
        to_date=to_date,
        token=token,
        max_cloud=max_cloud,
        limit=limit,
        tile=tile,
    )

    if not scenes:
        return {veg_index: [], "STR": [], "BOA": [], "SCL": [], "scenes": []}

    results = {veg_index: [], "STR": [], "BOA": [], "SCL": [], "scenes": []}

    for scene in scenes:
        scene_id = scene["id"]
        scene_datetime = scene["properties"]["datetime"]
        safe_time = scene_datetime.replace(":", "-").replace("/", "-")

        vi_path = folders["vi"] / f"{veg_index}_{safe_time}_{scene_id}.tif"
        str_path = folders["str"] / f"STR_{safe_time}_{scene_id}.tif"

        vi_file = download_index(
            aoi_geometry=aoi_geometry,
            scene_datetime=scene_datetime,
            script_name=veg_index,
            output_path=vi_path,
            token=token,
            swir_band=swir_band,
            width=width,
            height=height,
            overwrite=overwrite,
            scl_mask=scl_mask,
            scl_keep=scl_keep,
        )

        str_file = download_index(
            aoi_geometry=aoi_geometry,
            scene_datetime=scene_datetime,
            script_name="STR",
            output_path=str_path,
            token=token,
            swir_band=swir_band,
            width=width,
            height=height,
            overwrite=overwrite,
        )

        boa_file = None
        scl_file = None

        if not only_vi_str and "boa" in folders:
            boa_path = folders["boa"] / f"BOA_{safe_time}_{scene_id}.tif"
            boa_file = download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="BOA",
                output_path=boa_path,
                token=token,
                swir_band=swir_band,
                width=width,
                height=height,
                overwrite=overwrite,
            )

        if download_scl and "scl" in folders:
            scl_path = folders["scl"] / f"SCL_{safe_time}_{scene_id}.tif"
            scl_file = download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="SCL",
                output_path=scl_path,
                token=token,
                swir_band=swir_band,
                width=width,
                height=height,
                overwrite=overwrite,
            )

        results[veg_index].append(vi_file)
        results["STR"].append(str_file)
        if boa_file is not None:
            results["BOA"].append(boa_file)
        if scl_file is not None:
            results["SCL"].append(scl_file)

        results["scenes"].append(
            _scene_record(
                scene, veg_index, vi_file, str_file, boa_file, scl_file
            )
        )

    if scl_keep is not None:
        results["scl_keep"] = sorted(int(value) for value in set(scl_keep))

    return results
