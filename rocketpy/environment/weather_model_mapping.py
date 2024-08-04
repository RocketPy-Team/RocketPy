class WeatherModelMapping:
    """Class to map the weather model variables to the variables used in the
    Environment class.
    """

    GFS = {
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
    ECMWF = {
        "time": "time",
        "latitude": "latitude",
        "longitude": "longitude",
        "level": "level",
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

    def __init__(self):
        """Initialize the class, creates a dictionary with all the weather models
        available and their respective dictionaries with the variables."""

        self.all_dictionaries = {
            "GFS": self.GFS,
            "NAM": self.NAM,
            "ECMWF": self.ECMWF,
            "NOAA": self.NOAA,
            "RAP": self.RAP,
            "CMC": self.CMC,
            "GEFS": self.GEFS,
            "HIRESW": self.HIRESW,
        }

    def get(self, model):
        try:
            return self.all_dictionaries[model]
        except KeyError as e:
            raise KeyError(
                f"Model {model} not found in the WeatherModelMapping. "
                f"The available models are: {self.all_dictionaries.keys()}"
            ) from e
