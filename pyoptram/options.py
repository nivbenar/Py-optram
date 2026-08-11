### OPTRAM Package Options
# Stores and validates the session-level options implemented by pyOPTRAM,
# using rOPTRAM names and defaults.

from math import isfinite
from numbers import Real


_UNSET = object()


### Return whether a value is a finite, non-boolean real number.
def _numeric(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


### Build a validator that accepts one of the supplied values.
def _one_of(*values):
    return lambda value: value in values


### Return whether a value is a Python boolean.
def _boolean(value):
    return isinstance(value, bool)


### Validate rOPTRAM porosity, including its observable NA behavior.
def _porosity(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and (value != value or 0 < value < 1)
    )


_OPTION_SCHEMA = {
    "veg_index": ("NDVI", _one_of("NDVI", "SAVI", "MSAVI", "CI", "BSCI")),
    "max_cloud": (12, lambda value: _numeric(value) and 0 <= value <= 100),
    "vi_step": (0.005, lambda value: _numeric(value) and 0 < value <= 0.02),
    "trapezoid_method": (
        "linear",
        _one_of("linear", "exponential", "polynomial"),
    ),
    "SWIR_band": (11, _one_of(11, 12)),
    "max_tbl_size": (1_000_000, lambda value: _numeric(value) and value >= 10_000),
    "rm.low.vi": (False, _boolean),
    "rm.hi.str": (False, _boolean),
    "plot_colors": (
        "no",
        _one_of(
            "no",
            "none",
            "density",
            "feature",
            "features",
            "contour",
            "contours",
            "month",
            "months",
        ),
    ),
    "feature_col": ("ID", lambda value: isinstance(value, str)),
    "edge_points": (True, _boolean),
    "only_vi_str": (False, _boolean),
    "tileid": (
        None,
        lambda value: value is None or (isinstance(value, str) and len(value) == 5),
    ),
    "scm_mask": (True, _boolean),
    "overwrite": (False, _boolean),
    "resolution": (10, _one_of(10, 20, 60)),
    "porosity": (0.4, _porosity),
}

### Return a fresh mapping of implemented OPTRAM option defaults.
def _default_options():
    return {name: default for name, (default, _) in _OPTION_SCHEMA.items()}


_options = _default_options()


### Return the current value of one implemented OPTRAM option.
def get_optram_option(name):
    if name not in _OPTION_SCHEMA:
        raise ValueError(f"Unknown option name: {name}")
    return _options[name]


### Resolve an option-backed argument and validate explicit values.
def _resolve_optram_option(name, value=_UNSET, error_message=None):
    if name not in _OPTION_SCHEMA:
        raise ValueError(f"Unknown option name: {name}")
    if value is _UNSET:
        return _options[name]

    _, validator = _OPTION_SCHEMA[name]
    if not validator(value):
        message = error_message or f"Incorrect value: {value!r} for {name}"
        raise ValueError(message)
    return value


### Display, set, or reset rOPTRAM-compatible session options.
def optram_options(opt_name=None, opt_value=_UNSET, show_opts=True, reset=False):
    """Display, update, or reset implemented rOPTRAM-compatible options.

    Parameters
    ----------
    opt_name : str, optional
        Option to update. Omitting it leaves the current options unchanged.
    opt_value : object, optional
        New value, validated according to ``opt_name``. The implemented
        defaults match rOPTRAM.
    show_opts : bool, default True
        Print every current option after applying the requested operation.
    reset : bool, default False
        Restore all implemented options to their rOPTRAM defaults first.

    Returns
    -------
    dict
        A copy of the current implemented option values.

    Raises
    ------
    ValueError
        If the option name is unknown or its value fails validation.
    """
    if reset:
        _options.clear()
        _options.update(_default_options())

    if opt_name is not None and opt_value is not _UNSET:
        _options[opt_name] = _resolve_optram_option(opt_name, opt_value)

    current = dict(_options)
    if show_opts:
        for name, value in current.items():
            print(f"{name} = {value}")
    return current
