# era5_to_rocketpy_patch.py
# Author: Amanda Wech
# Mission: Make ERA5 NetCDF work with RocketPy EnvironmentAnalysis
# Status: LIFTOFF ACHIEVED

import xarray as xr
import numpy as np
import os

def create_era5_compatible_surface(input_path, output_path, target_lat, target_lon):
    """
    Patch ERA5 surface data to be RocketPy-compatible for RocketPy EnvironmentAnalysis.

    This function loads an ERA5 NetCDF surface file, renames and adds required variables,
    snaps the data to the nearest grid point to the target latitude and longitude, and
    writes the patched dataset to a new NetCDF file.

    Parameters
    ----------
    input_path : str
        Path to the input ERA5 NetCDF surface file.
    output_path : str
        Path to the output patched NetCDF file.
    target_lat : float
        Target latitude (in degrees) to snap to the nearest grid point.
    target_lon : float
        Target longitude (in degrees) to snap to the nearest grid point.

    Returns
    -------
    best_lat : float
        Latitude of the nearest grid point to the target latitude.
    best_lon : float
        Longitude of the nearest grid point to the target longitude.

    Raises
    ------
    FileNotFoundError
        If the input_path does not exist.
    KeyError
        If required variables (e.g., 'u10', 'latitude', 'longitude') are missing from the dataset.

    Example
    -------
    >>> best_lat, best_lon = create_era5_compatible_surface(
    ...     "input_surface.nc", "patched_surface.nc", -23.5, -46.6
    ... )
    >>> print(best_lat, best_lon)
    """
    print(f"Loading ERA5 surface: {input_path}")
    ds = xr.open_dataset(input_path)

    # Rename time
    if 'valid_time' in ds:
        ds = ds.rename({'valid_time': 'time'})
        print("   → Renamed 'valid_time' to 'time'")

    # Add v10, u100, v100
    if 'v10' not in ds:
        if 'u10' in ds:
            ds['v10'] = xr.zeros_like(ds['u10'])
            print("   → Added 'v10' (zeroed)")
        else:
            raise KeyError("Dataset is missing both 'v10' and 'u10'. Cannot create 'v10'.")
    ds['u100'] = ds['u10']
    ds['v100'] = ds['v10']
    print("   → Added u100/v100")

    # Required variables
    for var, val in {'t2m': 293.15, 'sp': 95000.0, 'tp': 0.0, 'z': 0.0}.items():
        if var not in ds:
            ds[var] = ds['u10'] * 0 + val
            print(f"   → Added '{var}' = {val}")

    # Optional variables
    for var, val in {'cbh': 1000.0, 'i10fg': 10.0, 'tcc': 0.0, 'blh': 500.0,
                     'sshf': 0.0, 'slhf': 0.0, 'ssrd': 0.0}.items():
        if var not in ds:
            ds[var] = ds['u10'] * 0 + val
            print(f"   → Added '{var}' = {val}")

    # Snap to grid
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    best_lat = float(lats[np.argmin(np.abs(lats - target_lat))])
    best_lon = float(lons[np.argmin(np.abs(lons - target_lon))])
    dist_km = ((target_lat - best_lat)**2 + (target_lon - best_lon)**2)**0.5 * 111
    print(f"   → Grid point: ({best_lat}, {best_lon}) | ~{dist_km:.1f} km")

    ds.to_netcdf(output_path)
    print(f"ERA5 SURFACE PATCHED → {output_path}")
    ds.close()
    return best_lat, best_lon


def create_dummy_pressure(output_path, best_lat, best_lon):
    """Create 2x2 dummy pressure file to avoid RocketPy index bugs."""
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted old: {output_path}")

    lat_vals = np.array([best_lat - 0.1, best_lat + 0.1])
    lon_vals = np.array([best_lon - 0.1, best_lon + 0.1])
    level_vals = np.array([1000, 850, 700, 500, 300, 200, 100])
    time_vals = np.array([], dtype='datetime64[h]')
    shape = (0, len(level_vals), 2, 2)
    empty_4d = np.full(shape, np.nan)

    ds = xr.Dataset(
        {
            'u': (['time', 'level', 'latitude', 'longitude'], empty_4d),
            'v': (['time', 'level', 'latitude', 'longitude'], empty_4d),
            't': (['time', 'level', 'latitude', 'longitude'], empty_4d),
            'z': (['time', 'level', 'latitude', 'longitude'], empty_4d),
        },
        coords={
            'time': ('time', time_vals),
            'level': ('level', level_vals, {'units': 'hPa'}),
            'latitude': ('latitude', lat_vals, {'units': 'degrees_north'}),
            'longitude': ('longitude', lon_vals, {'units': 'degrees_east'})
        }
    )
    ds.to_netcdf(output_path)
    print(f"DUMMY PRESSURE CREATED → {output_path}")
