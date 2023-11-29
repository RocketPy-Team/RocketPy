import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

necessary_require = [
    "numpy>=1.13",
    "scipy>=1.0",
    "matplotlib>=3.0",
    "netCDF4>=1.6.4",
    "requests",
    "pytz",
    "simplekml",
]

env_analysis_require = [
    "timezonefinder",
    "windrose>=1.6.8",
    "IPython",
    "ipywidgets>=7.6.3",
    "jsonpickle",
]

setuptools.setup(
    name="rocketpy",
    version="1.1.3",
    install_requires=necessary_require,
    extras_require={
        "env_analysis": env_analysis_require,
        "all": necessary_require + env_analysis_require,
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
    python_requires=">=3.8",
)
