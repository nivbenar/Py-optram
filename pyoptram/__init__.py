### Public Package API
# Re-exports the acquisition, transformation, fitting, and soil-moisture
# functions that form pyOPTRAM's supported interface.

from .acquire import (
    acquire_optram_inputs,
    get_cdse_token,
    retrieve_cdse_credentials,
    store_cdse_credentials,
    store_cdse_credentials_from_file,
)
from .ndvi_str import optram_ndvi_str
from .options import optram_options
from .soil_moisture import calculate_soil_moisture, optram_calculate_soil_moisture
from .str_transform_calculations import calculate_str, optram_calculate_str
from .trapezoid import optram_wetdry_coefficients, plot_vi_str_cloud
from .vegetation_index import calculate_vi

__all__ = [
    "acquire_optram_inputs",
    "get_cdse_token",
    "retrieve_cdse_credentials",
    "store_cdse_credentials",
    "store_cdse_credentials_from_file",
    "optram_ndvi_str",
    "optram_options",
    "calculate_soil_moisture",
    "optram_calculate_soil_moisture",
    "calculate_str",
    "optram_calculate_str",
    "optram_wetdry_coefficients",
    "plot_vi_str_cloud",
    "calculate_vi",
]
