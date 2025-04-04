import json
import os

import pytest
from scipy.stats import norm

from rocketpy import MultivariateRejectionSampler
from rocketpy._encoders import RocketPyEncoder


# pylint: disable=too-many-statements
def test_mrs_initialization():
    """Tests if the MultivariateRejectionSampler initialization opens input and output
    files correctly, and if it raises errors correctly when the files are problematic.
    """
    mrs_prefix = "mrs"

    # Tests if the input and output files opens correctly when the input is valid
    valid_mc_filepath_prefix = "valid_mc"
    valid_inputs = [
        {"a": 1, "b": 2, "c": [{"d": 1}]},
        {"a": 3, "b": 4, "c": [{"d": 1}]},
    ]
    valid_outputs = [
        {"e": 10, "f": 20},
        {"e": 30, "f": 40},
    ]
    with open(valid_mc_filepath_prefix + ".inputs.txt", "w+") as file:
        for json_input in valid_inputs:
            file.write(json.dumps(json_input, cls=RocketPyEncoder) + "\n")
    with open(valid_mc_filepath_prefix + ".outputs.txt", "w+") as file:
        for json_output in valid_outputs:
            file.write(json.dumps(json_output, cls=RocketPyEncoder) + "\n")
    MultivariateRejectionSampler(valid_mc_filepath_prefix, mrs_prefix)

    # tests if it raises an error when the file does not exist
    with pytest.raises(FileNotFoundError):
        MultivariateRejectionSampler("non_existent_mc_prefix", mrs_prefix)

    # tests if it raises an error when the input and output file contains
    # different number of samples
    invalid_mc_filepath_prefix = "invalid_mc"
    invalid_inputs = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    invalid_outputs = [{"c": 10, "d": 40}]
    with open(invalid_mc_filepath_prefix + ".inputs.txt", "w+") as file:
        for json_input in invalid_inputs:
            file.write(json.dumps(json_input, cls=RocketPyEncoder) + "\n")
    with open(invalid_mc_filepath_prefix + ".outputs.txt", "w+") as file:
        for json_output in invalid_outputs:
            file.write(json.dumps(json_output, cls=RocketPyEncoder) + "\n")
    with pytest.raises(ValueError):
        MultivariateRejectionSampler(invalid_mc_filepath_prefix, mrs_prefix)

    os.remove(valid_mc_filepath_prefix + ".inputs.txt")
    os.remove(valid_mc_filepath_prefix + ".outputs.txt")
    os.remove(invalid_mc_filepath_prefix + ".inputs.txt")
    os.remove(invalid_mc_filepath_prefix + ".outputs.txt")


def test_mrs_sample():
    """Tests if the MultivariateRejectionSampler samples correctly and raises errors
    when a non-existing variable is used in the distribution dict.
    """
    mrs_prefix = "mrs"

    # Tests if the input and output files opens correctly when the input is valid
    mc_filepath_prefix = "valid_mc"
    mc_inputs = [
        {"a": 0},
        {"a": 0.1},
        {"a": -0.1},
        {"a": 0.2},
        {"a": -0.2},
        {"a": 1},
        {"a": 1.1},
        {"a": -1.1},
        {"a": 1.2},
        {"a": -1.2},
    ]
    mc_outputs = [
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
        {"b": 10},
    ]
    with open(mc_filepath_prefix + ".inputs.txt", "w+") as file:
        for json_input in mc_inputs:
            file.write(json.dumps(json_input, cls=RocketPyEncoder) + "\n")
    with open(mc_filepath_prefix + ".outputs.txt", "w+") as file:
        for json_output in mc_outputs:
            file.write(json.dumps(json_output, cls=RocketPyEncoder) + "\n")
    mrs = MultivariateRejectionSampler(mc_filepath_prefix, mrs_prefix)

    invalid_distribution_dict = {"invalid_name": (norm(0, 1).pdf, norm(0, 1).pdf)}
    with pytest.raises(ValueError):
        mrs.sample(invalid_distribution_dict)
    valid_distribution_dict = {"a": (norm(0, 1).pdf, norm(0, 1).pdf)}
    mrs.sample(valid_distribution_dict)

    os.remove(mc_filepath_prefix + ".inputs.txt")
    os.remove(mc_filepath_prefix + ".outputs.txt")
    os.remove(mrs_prefix + ".inputs.txt")
    os.remove(mrs_prefix + ".outputs.txt")
    os.remove(mrs_prefix + ".errors.txt")
