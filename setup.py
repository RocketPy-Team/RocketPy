import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rocketpy",
    version="0.10.0",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=3.0",
        "requests",
        "pytz",
    ],
    maintainer="RocketPy Developers",
    author="Giovani Hidalgo Ceotto",
    author_email="ghceotto@gmail.com",
    description="Advanced 6-DOF trajectory simulation for High-Power Rocketry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giovaniceotto/RocketPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
