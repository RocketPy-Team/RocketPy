import os
import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Manage different netCDF4 versions depending on system version and Google Colab
netCDF4_requirement = "netCDF4>=1.6.2"
if sys.version_info[1] == 7:
    # Support for Python 3.7
    netCDF4_requirement = "netCDF4>=1.4,<1.6"

# Check if running on a linux system
elif sys.platform == "linux":
    if sys.version_info[1] < 11:
        # Support for Linux with Python < 3.11
        # Avoid problems due to netcdf4 issue linked below
        netCDF4_requirement = "netCDF4>=1.4,<1.6"
    else:
        # Support for Linux with Python >= 3.11
        # Might have problems with openDAP in the Environment class due to
        # https://github.com/Unidata/netcdf4-python/issues/1179
        netCDF4_requirement = "netCDF4>=1.6.2"

setuptools.setup(
    name="rocketpy",
    version="0.13.1",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=3.0",
        netCDF4_requirement,
        "windrose>=1.6.8",
        "ipywidgets>=7.6.3",
        "requests",
        "pytz",
        "simplekml",
        "jsonpickle",
    ],
    extras_require={
        "timezonefinder": ["timezonefinder"],
    },
    maintainer="RocketPy Developers",
    author="Giovani Hidalgo Ceotto",
    author_email="ghceotto@gmail.com",
    description="Advanced 6-DOF trajectory simulation for High-Power Rocketry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RocketPy-Team/RocketPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
