# Standard library
import importlib.util as iu
import pathlib
from dataclasses import dataclass, fields

# Third-party
import astropy.table as at
import astropy.units as u
import yaml
from astropy.time import Time

# Project
from .log import logger

__all__ = ["Config"]


@dataclass
class Config:
    name: str = None
    description: str = ""
    cache_path: (str, pathlib.Path) = None

    input_data_file: (str, pathlib.Path) = None
    input_data_format: str = None  # passed to astropy.table.Table.read()
    source_id_colname: str = None

    # Data specification
    rv_colname: str = None
    rv_error_colname: str = None
    rv_unit: str = "km/s"
    time_colname: str = None  # e.g., 'JD'
    time_format: str = "jd"  # passed to astropy.time.Time()
    time_scale: str = "tdb"  # passed to astropy.time.Time()

    # The Joker
    prior_file: (str, pathlib.Path) = "prior.py"
    n_prior_samples: int = None
    max_prior_samples: int = None
    prior_cache_file: (str, pathlib.Path) = None
    requested_samples_per_star: int = 1024
    randomize_prior_order: bool = False
    init_batch_size: int = None

    # Pipeline choices
    rerun_logP_ptp_threshold: float = 1.0
    rerun_P_factor: float = 2.5
    rerun_n_prior_samples: int = 10_000_000

    # MCMC
    mcmc_tune_steps: int = 1000
    mcmc_draw_steps: int = 1000
    mcmc_chains: int = 4
    mcmc_target_accept: float = 0.95
    mcmc_max_r_hat: float = 1.2

    # For expanding the samples into subdirectories:
    expand_subdir_column: str = None

    def __init__(self, filename):
        self._load_validate_config_values(filename)
        self._cache = {}

    def _load_validate_config_values(self, filename):
        filename = pathlib.Path(filename).expanduser().absolute()
        if not filename.exists():
            raise OSError(f"Config file {filename!s} does not exist.")

        with open(filename) as f:
            vals = yaml.safe_load(f.read())

        # Validate types:
        kw = {}
        for field in fields(self):
            default = field.default
            if field.name == "cache_path":
                default = filename.parent

            val = vals.get(field.name, None)
            if val is None:
                val = default

            if val is not None and (
                field.name.endswith("_file") or field.name.endswith("_path")
            ):
                val = pathlib.Path(val)

            kw[field.name] = val

        if kw["cache_path"] == filename.parent:
            logger.debug(
                "The cache path was not explicitly specified, so "
                "using the config file path as the cache path: "
                f"{kw['cache_path']!s}"
            )

        # Specialized defaults
        if kw["prior_cache_file"] is None:
            filename = f"prior_samples_{kw['n_prior_samples']}" f"_{kw['name']}.hdf5"
            kw["prior_cache_file"] = kw["cache_path"] / filename

        if kw["max_prior_samples"] is None:
            kw["max_prior_samples"] = kw["n_prior_samples"]

        if kw["init_batch_size"] is None:
            kw["init_batch_size"] = min(250_000, kw["n_prior_samples"])

        # Normalize paths:
        for k, v in kw.items():
            if isinstance(v, pathlib.Path):
                kw[k] = v.expanduser().absolute()

        # Validate:
        allowed_None_names = ["input_data_format"]
        for field in fields(self):
            val = kw[field.name]
            if not isinstance(val, field.type) and field.name not in allowed_None_names:
                msg = (
                    f"Config field '{field.name}' has type {type(val)}, "
                    f"but should be one of: {field.type}"
                )
                raise ValueError(msg)

            setattr(self, field.name, val)

    # ----------
    # File paths
    #
    @property
    def joker_results_file(self):
        return self.cache_path / "thejoker-samples.hdf5"

    @property
    def mcmc_results_file(self):
        return self.cache_path / "mcmc-samples.hdf5"

    @property
    def tasks_file(self):
        return self.cache_path / "tasks.hdf5"

    @property
    def metadata_file(self):
        return self.cache_path / "metadata.fits"

    @property
    def metadata_joker_file(self):
        return self.cache_path / "metadata-thejoker.fits"

    @property
    def metadata_mcmc_file(self):
        return self.cache_path / "metadata-mcmc.fits"

    @property
    def constant_results_file(self):
        return self.cache_path / "constant.fits"

    # ------------
    # Data loading
    #
    @property
    def data(self, **kwargs):
        if "data" not in self._cache:
            self._cache["data"] = at.Table.read(
                self.input_data_file, format=self.input_data_format
            )

        return self._cache["data"]

    def get_source_data(self, source_id):
        from thejoker.data import RVData

        visits = self.data[self.data[self.source_id_colname] == source_id]
        t = Time(
            visits[self.time_colname].astype("f8"),
            format=self.time_format,
            scale=self.time_scale,
        )
        rv_unit = u.Unit(self.rv_unit)

        rv = u.Quantity(visits[self.rv_colname].astype("f8"))
        rv_err = u.Quantity(visits[self.rv_error_colname].astype("f8"))

        if rv.unit == u.one:
            rv = rv * rv_unit
            rv_err = rv_err * rv_unit

        return RVData(t=t, rv=rv, rv_err=rv_err)

    def get_data_samples(self, source_id, mcmc=False):
        import h5py
        import thejoker as tj

        data = self.get_source_data(source_id)

        if not mcmc:
            with h5py.File(self.joker_results_file, "r") as results_f:
                samples = tj.JokerSamples.read(results_f[source_id])

        else:
            with h5py.File(self.mcmc_results_file, "r") as results_f:
                samples = tj.JokerSamples.read(results_f[source_id])

        try:
            MAP_sample = samples[samples["ln_likelihood"].argmax()]
        except KeyError:
            MAP_sample = None

        return data, samples, MAP_sample

    def get_prior(self, which=None, **kwargs):
        spec = iu.spec_from_file_location("prior", self.prior_file)
        user_prior = iu.module_from_spec(spec)
        spec.loader.exec_module(user_prior)
        if which == "mcmc":
            return user_prior.get_prior_mcmc(**kwargs)
        else:
            return user_prior.get_prior(**kwargs)

    def get_custom_init_mcmc(self):
        spec = iu.spec_from_file_location("prior", self.prior_file)
        user_prior = iu.module_from_spec(spec)
        spec.loader.exec_module(user_prior)
        return getattr(user_prior, "custom_init_mcmc", None)

    # ---------------
    # Special methods
    #
    def __getstate__(self):
        """Ensure that the cache does not get pickled with the object"""
        state = {k: v for k, v in self.__dict__.items() if k != "_cache"}
        state["_cache"] = {}
        return state.copy()
