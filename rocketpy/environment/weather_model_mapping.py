class WeatherModelMapping:
    """Map provider-specific variable names to RocketPy weather fields.

    RocketPy reads forecast/reanalysis/ensemble datasets using canonical keys
    such as ``time``, ``latitude``, ``longitude``, ``level``, ``temperature``,
    ``geopotential_height``, ``geopotential``, ``u_wind`` and ``v_wind``.
    Each dictionary in this class maps those canonical keys to the actual
    variable names in a specific data provider format.

    Mapping families
    ----------------
    - Base names (for example ``GFS``, ``NAM``, ``RAP``) represent the current
      default mappings used by the latest-model shortcuts and THREDDS-style
      datasets.
    - ``*_LEGACY`` names represent older NOMADS-style variable naming
      conventions (for example ``lev``, ``tmpprs``, ``ugrdprs`` and
      ``vgrdprs``) and are intended for archived or previously downloaded files.

    Notes
    -----
    - Mappings can also include optional keys such as ``projection`` for
      projected grids and ``ensemble`` for member dimensions.
    - The :meth:`get` method is case-insensitive, so ``"gfs_legacy"`` and
      ``"GFS_LEGACY"`` are equivalent.
    """

    GFS = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "isobaric",
        "temperature": "Temperature_isobaric",
        "surface_geopotential_height": "Geopotential_height_surface",
        "geopotential_height": "Geopotential_height_isobaric",
        "geopotential": None,
        "u_wind": "u-component_of_wind_isobaric",
        "v_wind": "v-component_of_wind_isobaric",
    }
    GFS_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    NAM = {
        "time": "time",
        "latitude": "y",
        "longitude": "x",
        "projection": "LambertConformal_Projection",
        "level": "isobaric",
        "temperature": "Temperature_isobaric",
        "surface_geopotential_height": None,
        "geopotential_height": "Geopotential_height_isobaric",
        "geopotential": None,
        "u_wind": "u-component_of_wind_isobaric",
        "v_wind": "v-component_of_wind_isobaric",
    }
    NAM_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    ECMWF_v0 = {
        "time": "time",
        "latitude": "latitude",
        "longitude": "longitude",
        "level": "level",
        "ensemble": "number",
        "temperature": "t",
        "surface_geopotential_height": None,
        "geopotential_height": None,
        "geopotential": "z",
        "u_wind": "u",
        "v_wind": "v",
    }
    ECMWF = {
        "time": "valid_time",
        "latitude": "latitude",
        "longitude": "longitude",
        "level": "pressure_level",
        "ensemble": "number",
        "temperature": "t",
        "surface_geopotential_height": None,
        "geopotential_height": None,
        "geopotential": "z",
        "u_wind": "u",
        "v_wind": "v",
    }
    NOAA = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "isobaric",
        "temperature": "Temperature_isobaric",
        "surface_geopotential_height": "Geopotential_height_surface",
        "geopotential_height": "Geopotential_height_isobaric",
        "geopotential": None,
        "u_wind": "u-component_of_wind_isobaric",
        "v_wind": "v-component_of_wind_isobaric",
    }
    NOAA_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    RAP = {
        "time": "time",
        "latitude": "y",
        "longitude": "x",
        "projection": "LambertConformal_Projection",
        "level": "isobaric",
        "temperature": "Temperature_isobaric",
        "surface_geopotential_height": None,
        "geopotential_height": "Geopotential_height_isobaric",
        "geopotential": None,
        "u_wind": "u-component_of_wind_isobaric",
        "v_wind": "v-component_of_wind_isobaric",
    }
    RAP_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    CMC = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "ensemble": "ens",
        "temperature": "tmpprs",
        "surface_geopotential_height": None,
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    CMC_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "ensemble": "ens",
        "temperature": "tmpprs",
        "surface_geopotential_height": None,
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    GEFS = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "ensemble": "ens",
        "temperature": "tmpprs",
        "surface_geopotential_height": None,
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    GEFS_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "ensemble": "ens",
        "temperature": "tmpprs",
        "surface_geopotential_height": None,
        "geopotential_height": "hgtprs",
        "geopotential": None,
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    HIRESW = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    HIRESW_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "tmpprs",
        "surface_geopotential_height": "hgtsfc",
        "geopotential_height": "hgtprs",
        "u_wind": "ugrdprs",
        "v_wind": "vgrdprs",
    }
    MERRA2 = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "T",
        "surface_geopotential_height": None,
        "surface_geopotential": "PHIS",  # special key for Geopotential (m^2/s^2)
        "geopotential_height": "H",
        "geopotential": None,
        "u_wind": "U",
        "v_wind": "V",
    }
    MERRA2_LEGACY = {
        "time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "level": "lev",
        "temperature": "T",
        "surface_geopotential_height": None,
        "surface_geopotential": "PHIS",
        "geopotential_height": "H",
        "geopotential": None,
        "u_wind": "U",
        "v_wind": "V",
    }

    def __init__(self):
        """Build the lookup table with default and legacy mapping aliases."""

        self.all_dictionaries = {
            "GFS": self.GFS,
            "GFS_LEGACY": self.GFS_LEGACY,
            "NAM": self.NAM,
            "NAM_LEGACY": self.NAM_LEGACY,
            "ECMWF_v0": self.ECMWF_v0,
            "ECMWF": self.ECMWF,
            "NOAA": self.NOAA,
            "NOAA_LEGACY": self.NOAA_LEGACY,
            "RAP": self.RAP,
            "RAP_LEGACY": self.RAP_LEGACY,
            "CMC": self.CMC,
            "CMC_LEGACY": self.CMC_LEGACY,
            "GEFS": self.GEFS,
            "GEFS_LEGACY": self.GEFS_LEGACY,
            "HIRESW": self.HIRESW,
            "HIRESW_LEGACY": self.HIRESW_LEGACY,
            "MERRA2": self.MERRA2,
            "MERRA2_LEGACY": self.MERRA2_LEGACY,
        }

    def get(self, model):
        """Return a mapping dictionary by model alias (case-insensitive).

        Parameters
        ----------
        model : str
            Mapping alias name, such as ``"GFS"`` or ``"GFS_LEGACY"``.

        Returns
        -------
        dict
            Dictionary mapping RocketPy canonical weather keys to dataset
            variable names.

        Raises
        ------
        KeyError
            If ``model`` is unknown or not a string.
        """
        if not isinstance(model, str):
            raise KeyError(
                f"Model {model} not found in the WeatherModelMapping. "
                f"The available models are: {self.all_dictionaries.keys()}"
            )

        try:
            return self.all_dictionaries[model]
        except KeyError as exc:
            model_casefold = model.casefold()
            for key, value in self.all_dictionaries.items():
                if key.casefold() == model_casefold:
                    return value

            raise KeyError(
                f"Model {model} not found in the WeatherModelMapping. "
                f"The available models are: {self.all_dictionaries.keys()}"
            ) from exc
