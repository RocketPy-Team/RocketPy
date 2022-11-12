import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rocketpy",
    version="0.12.0",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=3.0",
        "netCDF4>=1.4",
        "windrose>=1.6.8",
        "requests",
        "pytz",
        "timezonefinder",
        "simplekml",
    ],
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
    python_requires=">=3.6",
)
