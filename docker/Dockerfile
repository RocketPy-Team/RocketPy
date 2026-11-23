# Set base image 
# python:3.14 is the newest version the test matrix covers, and is what the
# python:latest tag resolves to today. Pinning keeps the two from drifting apart.
# Get a full list of python images here: https://hub.docker.com/_/python/tags
FROM python:3.14

# set the working directory in the container
WORKDIR /RocketPy

# Ensure pip is updated
RUN python3 -m pip install --upgrade pip

# Copy the dependencies file to the working directory
COPY requirements.txt .
COPY requirements-tests.txt .

# Install dependencies
# Use a single RUN instruction to minimize the number of layers
RUN pip install \
    -r requirements.txt \
    -r requirements-tests.txt

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
# print the operational system and the python version
CMD [ "python3", "-c", "import platform;import sys; print('Python ', sys.version, ' running on ', platform.platform())" ]

# Install the rocketpy package # TODO: check if I can put this in editable mode
RUN pip install .
