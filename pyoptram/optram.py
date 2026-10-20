"""High-level orchestration matching the supported CDSE path of rOPTRAM::optram."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from .acquire import acquire_optram_inputs
from .ndvi_str import optram_ndvi_str
from .options import get_optram_option
from .trapezoid import optram_wetdry_coefficients, plot_vi_str_cloud


def optram(
    aoi,
    from_date,
    to_date,
    s2_output_dir=None,
    data_output_dir=None,
):
    """Acquire OPTRAM inputs, build the VI-STR table, and fit its edges.

    This is the Python counterpart of the default CDSE path orchestrated by
    ``rOPTRAM::optram()``. Omitted output directories use the system temporary
    directory. Acquisition and scientific options remain owned by the child
    functions and :func:`pyoptram.optram_options`.

    The wrapper saves ``vi_str_trapezoid.png`` in ``data_output_dir``, then
    prints and returns the wet/dry fitting RMSE dataframe. It does not calculate
    soil moisture, matching the R wrapper. Automatic plot saving is a
    Py-optram enhancement; rOPTRAM creates its plot separately. rOPTRAM's
    optional openEO backend is not yet implemented in Py-optram.

    Parameters
    ----------
    aoi : geopandas.GeoDataFrame
        Polygon or MultiPolygon area of interest. The original features are
        forwarded to :func:`optram_ndvi_str` for optional feature labeling.
    from_date, to_date : str
        Acquisition date range formatted as ``YYYY-MM-DD``.
    s2_output_dir : path-like, optional
        Directory for acquired VI/STR imagery.
    data_output_dir : path-like, optional
        Directory for the VI-STR Parquet table, wet/dry fitting outputs, and
        ``vi_str_trapezoid.png``.

    Returns
    -------
    pandas.DataFrame
        One-row dataframe containing wet and dry edge RMSE values.
    """
    if s2_output_dir is None:
        s2_output_dir = tempfile.gettempdir()
    if data_output_dir is None:
        data_output_dir = tempfile.gettempdir()

    acquire_optram_inputs(
        aoi=aoi,
        from_date=from_date,
        to_date=to_date,
        output_dir=s2_output_dir,
    )

    veg_index = get_optram_option("veg_index")
    s2_output_dir = Path(s2_output_dir)
    vi_paths = sorted((s2_output_dir / veg_index).iterdir())
    str_paths = sorted((s2_output_dir / "STR").iterdir())

    vi_str = optram_ndvi_str(
        vi_paths,
        str_paths,
        output_parquet=Path(data_output_dir) / "VI_STR_data.parquet",
        features=aoi,
    )
    rmse_df, _, edges_df = optram_wetdry_coefficients(
        vi_str,
        output_dir=data_output_dir,
        return_outputs=True,
    )

    plot_path = Path(data_output_dir) / "vi_str_trapezoid.png"
    plot_ax = plot_vi_str_cloud(
        vi_str,
        edges_df,
        output_path=plot_path,
    )
    plt.close(plot_ax.figure)
    print(f"Trapezoid plot saved to: {plot_path}")

    print("RMSE for fitted trapezoid:")
    print(rmse_df)
    return rmse_df
