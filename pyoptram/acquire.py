### Copernicus Data Space Acquisition
# Stores CDSE credentials, searches the Sentinel-2 L2A catalog, and downloads
# vegetation-index, STR, and BOA rasters for OPTRAM workflows.

from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path
import csv
import json
import math
import os
import platform
import warnings

import geopandas as gpd
import requests
import s2rst
from shapely.geometry import mapping, shape

from .options import _UNSET, _resolve_optram_option, get_optram_option


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEFAULT_MAX_SIZE = 2500


### Calculate metre lengths of longitude and latitude degrees on WGS 84.
def _degree_lengths(latitude):
    """Return CDSE-compatible metre lengths of one degree at ``latitude``."""
    a = 6378137.0
    b = 6356752.3142
    e2 = (a ** 2 - b ** 2) / a ** 2
    phi = latitude * math.pi / 180.0
    denominator = 1 - e2 * math.sin(phi) ** 2
    degree_x = (
        math.pi * a * math.cos(phi) / (180.0 * math.sqrt(denominator))
    )
    degree_y = (
        math.pi * a * (1 - e2) / (180.0 * denominator ** (3 / 2))
    )
    return degree_x, degree_y


### Build the Process API grid used by CDSE for metre-resolution requests.
def _resolution_output(aoi_geometry, resolution):
    """Convert metres to a CRS84 grid and enforce CDSE's 2500-pixel limit."""
    geometry = shape(aoi_geometry)
    centroid_degree_lengths = _degree_lengths(geometry.centroid.y)
    resx = resolution / centroid_degree_lengths[0]
    resy = resolution / centroid_degree_lengths[1]

    minx, miny, maxx, maxy = geometry.bounds
    midpoint_degree_lengths = _degree_lengths((miny + maxy) / 2)
    check_resx = resolution / midpoint_degree_lengths[0]
    check_resy = resolution / midpoint_degree_lengths[1]
    columns = max(1, math.floor((maxx - minx) / check_resx + 0.5))
    rows = max(1, math.floor((maxy - miny) / check_resy + 0.5))
    if rows > DEFAULT_MAX_SIZE or columns > DEFAULT_MAX_SIZE:
        raise ValueError(
            f"The requested image dimension ({rows} x {columns}) exceeds "
            "the allowed maximum (2500 pixels). Please revise the "
            "resolution to make sure it is in supported range."
        )
    return {"resx": resx, "resy": resy}


### Validate explicit Python-specific Process API dimensions.
def _dimension_output(width, height):
    if width is None:
        width = DEFAULT_MAX_SIZE
    if height is None:
        height = DEFAULT_MAX_SIZE
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width < 1
        or height < 1
    ):
        raise ValueError("width and height must be positive integers")
    if width > DEFAULT_MAX_SIZE or height > DEFAULT_MAX_SIZE:
        raise ValueError("width and height cannot exceed 2500 pixels")
    return {"width": width, "height": height}


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
    """Store CDSE OAuth credentials in rOPTRAM-compatible local storage.

    Parameters
    ----------
    client_id, client_secret : str, optional
        OAuth credentials. If either is omitted, both are read from the
        ``OAUTH_CLIENTID`` and ``OAUTH_SECRET`` environment variables.

    Returns
    -------
    None
        Credentials are written as JSON under the platform-specific CDSE
        directory used by rOPTRAM. Nothing is written when a complete pair is
        unavailable; an unsupported platform emits ``RuntimeWarning``.
    """
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
    """Store CDSE OAuth credentials from a one-record CSV file.

    Parameters
    ----------
    path : path-like
        CSV containing ``clientid`` and ``secret`` headers and exactly one
        nonempty credential record.

    Returns
    -------
    None
        The record is saved through :func:`store_cdse_credentials`.

    Raises
    ------
    ValueError
        If headers, record count, or credential values are invalid.
    """
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
    """Retrieve CDSE credentials stored by Py-optram or rOPTRAM.

    Returns
    -------
    dict or None
        ``{"clientid": ..., "secret": ...}`` from the shared
        platform-specific JSON file, or ``None`` when usable credentials are
        unavailable. Missing storage and unsupported platforms emit
        ``RuntimeWarning``.
    """
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
    """Request a Copernicus Data Space access token.

    Parameters
    ----------
    client_id, client_secret : str
        CDSE OAuth client credentials.

    Returns
    -------
    str
        The ``access_token`` returned by the CDSE identity service.

    Raises
    ------
    requests.HTTPError
        If authentication fails. The exception reports the HTTP status but
        does not expose the supplied secret.
    """
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


### Validate and convert a GeoDataFrame AOI to one WGS84 GeoJSON geometry.
def load_aoi(aoi):
    """Return the union of a polygon GeoDataFrame in EPSG:4326."""
    if not isinstance(aoi, gpd.GeoDataFrame):
        raise TypeError("aoi must be a geopandas.GeoDataFrame")
    if aoi.empty or aoi.geometry.is_empty.any():
        raise ValueError("aoi must contain at least one non-empty geometry")
    geometry_types = aoi.geometry.geom_type
    if not ((geometry_types == "Polygon").all() or
            (geometry_types == "MultiPolygon").all()):
        raise ValueError("aoi must contain only Polygon or only MultiPolygon geometries")
    if aoi.crs is None:
        raise ValueError("aoi must have a CRS")

    aoi_wgs84 = aoi.to_crs(4326)
    return mapping(aoi_wgs84.geometry.union_all())


### Create output folders for VI, STR, and optional BOA rasters.
def prepare_output_folders(output_dir, veg_index=_UNSET, only_vi_str=_UNSET):
    """Create acquisition output directories for requested products.

    ``veg_index`` and ``only_vi_str`` default to their current package
    options, initially ``"NDVI"`` and ``False``. VI and STR directories are
    always created; BOA is omitted when ``only_vi_str`` is true.

    Returns
    -------
    dict
        Product keys mapped to created :class:`pathlib.Path` directories.
    """
    if veg_index is _UNSET:
        veg_index = get_optram_option("veg_index")
    if only_vi_str is _UNSET:
        only_vi_str = get_optram_option("only_vi_str")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = {"vi": output_dir / veg_index, "str": output_dir / "STR"}

    if not only_vi_str:
        folders["boa"] = output_dir / "BOA"

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


### Sentinel Hub scripts and catalog operations

### Return a packaged rOPTRAM Sentinel Hub evalscript.
def load_evalscript(script_name, swir_band=_UNSET, scl_mask=False):
    """Return the selected packaged rOPTRAM Process API evalscript."""
    if swir_band is _UNSET:
        swir_band = get_optram_option("SWIR_band")
    if script_name in {"NDVI", "SAVI", "MSAVI"}:
        suffix = "_masked" if scl_mask else ""
        script_file = f"{script_name}{suffix}.js"
    elif script_name == "STR":
        script_file = f"STR{swir_band}.js"
    elif script_name == "BOA":
        script_file = "BOA.js"
    else:
        raise ValueError(f"Unknown script_name: {script_name}")

    return (files("pyoptram") / "evalscripts" / script_file).read_text(
        encoding="utf-8"
    )


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

    return [scene for scene in scenes if tile in scene.get("id", "")]


### Convert one GeoJSON ring to a normalized S2 loop.
def _s2_loop(coordinates):
    if coordinates and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]
    loop = s2rst.Loop(
        [
            s2rst.LatLng.from_degrees(latitude, longitude).to_point()
            for longitude, latitude in coordinates
        ]
    )
    loop.normalize()
    return loop


### Convert a GeoJSON Polygon or MultiPolygon to an S2 polygon.
def _s2_polygon(geometry):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError("areaCoverage requires a Polygon or MultiPolygon")

    loops = [_s2_loop(ring) for polygon in polygons for ring in polygon]
    polygon = s2rst.Polygon(loops)
    if error := polygon.validate():
        raise ValueError(f"Invalid S2 polygon: {error}")
    return polygon


### Calculate rOPTRAM-style spherical AOI coverage, rounded to three decimals.
def _area_coverage(aoi_polygon, scene_geometry):
    scene_polygon = _s2_polygon(scene_geometry)
    aoi_index = s2rst.ShapeIndex()
    aoi_index.add(aoi_polygon)
    aoi_index.build()
    scene_index = s2rst.ShapeIndex()
    scene_index.add(scene_polygon)
    scene_index.build()
    intersection = s2rst.boolean_operation(
        s2rst.OpType.INTERSECTION, aoi_index, scene_index
    )
    return round(100 * intersection.area() / aoi_polygon.area(), 3)


### Return a catalog scene's UTC acquisition date.
def _scene_date(scene):
    timestamp = scene.get("properties", {}).get("datetime", "")
    return datetime.strptime(timestamp[:10], "%Y-%m-%d").date()


### Reproduce CDSE::SeasonalTimerange and SeasonalFilter.
def _seasonal_filter(scenes, from_date, to_date):
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    same_year_end = start.replace(month=end.month, day=end.day)
    crosses_year = same_year_end < start
    last_year = end.year - 1 if crosses_year else end.year
    ranges = []
    for year in range(start.year, last_year + 1):
        season_start = start.replace(year=year)
        season_end = end.replace(year=year + int(crosses_year))
        ranges.append((season_start, season_end))

    selected = [
        scene
        for scene in scenes
        if any(first <= _scene_date(scene) <= last for first, last in ranges)
    ]
    return [
        scene
        for _, scene in sorted(
            enumerate(selected),
            key=lambda item: (_scene_date(item[1]), item[0]),
            reverse=True,
        )
    ]


### Search the Sentinel-2 L2A catalog for matching scenes.
def search_catalog(
    aoi_geometry,
    from_date,
    to_date,
    token,
    max_cloud=_UNSET,
    limit=None,
    tile=_UNSET,
    area_cover=_UNSET,
    period=_UNSET,
):
    """Search and filter the Sentinel-2 L2A catalog like rOPTRAM/CDSE.

    Parameters
    ----------
    aoi_geometry : dict
        GeoJSON Polygon or MultiPolygon used as the spatial intersection.
    from_date, to_date : str
        Inclusive catalog timestamps expressed as validated ``YYYY-MM-DD``
        dates. Acquisition requires ``to_date > from_date``.
    token : str
        CDSE access token.
    max_cloud : float, optional
        Strict upper cloud-cover bound. Defaults to the rOPTRAM-compatible
        ``max_cloud`` option, initially 12.
    limit : int or None, default None
        Optional Python compatibility cap applied after parity filtering.
    tile : str or None, optional
        Optional five-character MGRS tile identifier.
    area_cover : float, optional
        Minimum spherical AOI coverage after three-decimal rounding. Defaults
        to 99.0.
    period : {"full", "seasonal"}, optional
        Full date range or rOPTRAM seasonal filtering. Defaults to ``"full"``.

    Returns
    -------
    list of dict
        Catalog feature records, after optional tile filtering.

    Raises
    ------
    ValueError
        If ``max_cloud`` is outside 0--100 or is not finite numeric data.
    requests.HTTPError
        If the catalog request fails.
    """
    max_cloud = _resolve_optram_option(
        "max_cloud",
        max_cloud,
        "max_cloud must be numeric and between 0 and 100",
    )
    tile = _resolve_optram_option(
        "tileid", tile, "tile must be None or a five-character MGRS tile ID"
    )
    area_cover = _resolve_optram_option(
        "area_cover",
        area_cover,
        "area_cover must be numeric and between 0 and 100",
    )
    period = _resolve_optram_option(
        "period", period, "period must be 'full' or 'seasonal'"
    )
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "intersects": aoi_geometry,
        "datetime": f"{from_date}T00:00:00Z/{to_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 100,
    }

    scenes = []
    while True:
        response = requests.post(
            CATALOG_URL, headers=headers, json=payload, timeout=60
        )
        _raise_for_status(response, "Sentinel-2 catalog search")
        page = response.json()
        scenes.extend(page.get("features", []))
        next_page = page.get("context", {}).get("next")
        if next_page is None:
            break
        payload["next"] = next_page

    scenes = [
        scene
        for scene in scenes
        if scene.get("properties", {}).get("eo:cloud_cover", math.inf) < max_cloud
    ]
    scenes = _filter_scenes_by_tile(scenes, tile)
    aoi_polygon = _s2_polygon(aoi_geometry)
    covered = []
    for scene in scenes:
        coverage = _area_coverage(aoi_polygon, scene["geometry"])
        scene = dict(scene)
        scene["areaCoverage"] = coverage
        if coverage >= area_cover:
            covered.append(scene)
    scenes = covered
    if period == "seasonal":
        scenes = _seasonal_filter(scenes, from_date, to_date)
    return scenes if limit is None else scenes[:limit]


### Download one GeoTIFF product through the Sentinel Hub Process API.
def download_index(
    aoi_geometry,
    scene_datetime,
    script_name,
    output_path,
    token,
    swir_band=_UNSET,
    resolution=_UNSET,
    width=None,
    height=None,
    overwrite=_UNSET,
    scl_mask=False,
):
    """Download one VI, STR, or BOA GeoTIFF through the Process API.

    Parameters
    ----------
    aoi_geometry : dict
        GeoJSON acquisition geometry.
    scene_datetime : str
        Catalog scene timestamp whose whole UTC day is used for the Process
        API time range, matching rOPTRAM's ``as.Date`` behavior.
    script_name : {"NDVI", "SAVI", "MSAVI", "STR", "BOA"}
        Product evalscript to run.
    output_path : path-like
        Destination GeoTIFF.
    token : str
        CDSE access token.
    swir_band : {11, 12}, optional
        STR band; defaults to the current option, initially 11.
    resolution : {10, 20, 60}, optional
        Output resolution in metres. The default follows rOPTRAM/CDSE and is
        converted to a CRS84 angular grid at the AOI centroid latitude.
    width, height : int, optional
        Python-specific Process API dimension overrides. Supplying either
        selects dimension mode; an omitted dimension defaults to 2500.
    overwrite : bool, optional
        Defaults to the current rOPTRAM-compatible option, initially false.
    scl_mask : bool, default False
        Apply SCL masking to a supported VI evalscript.

    Returns
    -------
    str
        Output path. An existing file is returned without downloading unless
        ``overwrite`` is true.

    Raises
    ------
    ValueError
        If ``script_name`` is unsupported.
    requests.HTTPError
        If the Process API request fails.
    """
    if swir_band is _UNSET:
        swir_band = get_optram_option("SWIR_band")
    resolution = _resolve_optram_option(
        "resolution", resolution, "resolution must be 10, 20, or 60"
    )
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

    if width is None and height is None:
        spatial_output = _resolution_output(aoi_geometry, resolution)
    else:
        spatial_output = _dimension_output(width, height)
    scene_day = scene_datetime[:10]
    next_day = (
        datetime.strptime(scene_day, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    payload = {
        "input": {
            "bounds": {
                "geometry": aoi_geometry,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{scene_day}T00:00:00.000Z",
                            "to": f"{next_day}T00:00:00.000Z",
                        },
                        "mosaickingOrder": "mostRecent",
                    },
                }
            ],
        },
        "output": {
            **spatial_output,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": load_evalscript(
            script_name,
            swir_band=swir_band,
            scl_mask=scl_mask,
        ),
    }

    response = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=180)
    _raise_for_status(response, f"{script_name} download")

    output_path.write_bytes(response.content)
    return str(output_path)


### Build a metadata record for one downloaded scene.
def _scene_record(scene, veg_index, vi_path, str_path, boa_path=None):
    properties = scene.get("properties", {})
    record = {
        "id": scene.get("id"),
        "datetime": properties.get("datetime"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "areaCoverage": scene.get("areaCoverage"),
        "tile": _scene_tile(scene),
        "STR": str_path,
        "BOA": boa_path,
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
    limit=None,
    resolution=_UNSET,
    width=None,
    height=None,
    overwrite=_UNSET,
    scl_mask=_UNSET,
    area_cover=_UNSET,
    period=_UNSET,
    save_img_list=_UNSET,
):
    """Acquire Sentinel-2 VI and STR inputs from Copernicus Data Space.

    Parameters
    ----------
    aoi : geopandas.GeoDataFrame
        Polygon or MultiPolygon area of interest. Multiple features are
        geometrically unioned before acquisition, matching rOPTRAM.
    from_date, to_date : str
        ``YYYY-MM-DD`` range with ``to_date`` strictly later than
        ``from_date``.
    output_dir : path-like
        Parent directory for VI, STR, and optional BOA rasters.
    client_id, client_secret : str, optional
        Explicit OAuth credentials. A stored pair is used if either is
        omitted.
    save_creds : bool, default True
        Save an explicit credential pair only after successful authentication.
    veg_index : {"NDVI", "SAVI", "MSAVI"}, optional
        Acquired VI. Defaults to the current ``veg_index`` option, initially
        ``"NDVI"``. CI and BSCI have no rOPTRAM CDSE evalscripts.
    swir_band : {11, 12}, optional
        STR band; defaults to 11 through ``SWIR_band``.
    max_cloud : float, optional
        Strict catalog cloud-cover bound; defaults to 12.
    only_vi_str : bool, optional
        Skip BOA downloads when true. Defaults to false.
    tile : str or None, optional
        Five-character MGRS tile; defaults to the ``tileid`` option.
    limit : int or None, default None
        Optional Python compatibility cap. By default all catalog pages are
        consumed and no scene-count cap is applied.
    resolution : {10, 20, 60}, optional
        Output resolution in metres; defaults to the current ``resolution``
        option, initially 10. As in rOPTRAM/CDSE, metres are converted to a
        CRS84 angular grid using the AOI centroid latitude.
    width, height : int, optional
        Python-specific output-dimension overrides, each limited to 2500.
        Supplying either selects dimension mode; an omitted value uses 2500.
    overwrite : bool, optional
        Redownload existing outputs; defaults to false.
    scl_mask : bool, optional
        Select the rOPTRAM VI evalscript retaining SCL classes 2, 4, 5, and
        10; defaults to true. STR remains unmasked.
    area_cover : float, optional
        Minimum spherical AOI coverage, rounded to three decimals before an
        inclusive comparison. Defaults to 99.0.
    period : {"full", "seasonal"}, optional
        Use the full date range or rOPTRAM's repeating seasonal windows.
        Defaults to ``"full"``.
    save_img_list : bool, optional
        Save the post-filter catalog before downloads. Defaults to false.

    Returns
    -------
    dict
        Lists keyed by the selected VI, ``STR``, ``BOA``, and ``scenes``.
        Scene records contain paths, timestamp, tile, cloud cover, and catalog
        ID.

    Raises
    ------
    ValueError
        For invalid options, dimensions, dates, unsupported acquisition VIs,
        or unavailable credentials.
    requests.HTTPError
        If authentication, catalog search, or a download fails.

    Notes
    -----
    This implements rOPTRAM's CDSE/Sentinel Hub path only. Its openEO workflow
    is not implemented here. Spherical area coverage uses the
    S2-style ``s2rst`` implementation. It can differ from R sf/S2 at artificial
    floating-point rounding boundaries by tiny amounts; three-decimal
    practical parity is the supported policy.
    """
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
    area_cover = _resolve_optram_option(
        "area_cover",
        area_cover,
        "area_cover must be numeric and between 0 and 100",
    )
    period = _resolve_optram_option(
        "period", period, "period must be 'full' or 'seasonal'"
    )
    save_img_list = _resolve_optram_option(
        "save_img_list", save_img_list, "save_img_list must be a boolean"
    )
    overwrite = _resolve_optram_option(
        "overwrite", overwrite, "overwrite must be a boolean"
    )
    scl_mask = _resolve_optram_option(
        "scm_mask", scl_mask, "scl_mask must be a boolean"
    )
    resolution = _resolve_optram_option(
        "resolution", resolution, "resolution must be 10, 20, or 60"
    )

    if veg_index not in ("NDVI", "SAVI", "MSAVI"):
        raise ValueError("CDSE acquisition supports NDVI, SAVI, or MSAVI")

    from_date = validate_date(from_date, "from_date")
    to_date = validate_date(to_date, "to_date")

    if from_date >= to_date:
        raise ValueError("to_date must be later than from_date")

    aoi_geometry = load_aoi(aoi)
    if width is None and height is None:
        _resolution_output(aoi_geometry, resolution)
    else:
        _dimension_output(width, height)

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
    )

    scenes = search_catalog(
        aoi_geometry=aoi_geometry,
        from_date=from_date,
        to_date=to_date,
        token=token,
        max_cloud=max_cloud,
        tile=tile,
        area_cover=area_cover,
        period=period,
        limit=limit,
    )

    if not scenes:
        return {veg_index: [], "STR": [], "BOA": [], "scenes": []}

    if save_img_list:
        image_list = {"type": "FeatureCollection", "features": scenes}
        (Path(output_dir) / "image_list.json").write_text(
            json.dumps(image_list, indent=2), encoding="utf-8"
        )

    results = {veg_index: [], "STR": [], "BOA": [], "scenes": []}

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
            resolution=resolution,
            width=width,
            height=height,
            overwrite=overwrite,
            scl_mask=scl_mask,
        )

        str_file = download_index(
            aoi_geometry=aoi_geometry,
            scene_datetime=scene_datetime,
            script_name="STR",
            output_path=str_path,
            token=token,
            swir_band=swir_band,
            resolution=resolution,
            width=width,
            height=height,
            overwrite=overwrite,
        )

        boa_file = None
        if not only_vi_str and "boa" in folders:
            boa_path = folders["boa"] / f"BOA_{safe_time}_{scene_id}.tif"
            boa_file = download_index(
                aoi_geometry=aoi_geometry,
                scene_datetime=scene_datetime,
                script_name="BOA",
                output_path=boa_path,
                token=token,
                swir_band=swir_band,
                resolution=resolution,
                width=width,
                height=height,
                overwrite=overwrite,
            )

        results[veg_index].append(vi_file)
        results["STR"].append(str_file)
        if boa_file is not None:
            results["BOA"].append(boa_file)
        results["scenes"].append(
            _scene_record(scene, veg_index, vi_file, str_file, boa_file)
        )

    return results
