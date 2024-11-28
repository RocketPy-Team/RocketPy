"""
Multivariate Rejection Sampling Module for RocketPy

Notes
-----
This module is still under active development, and some features or attributes may
change in future versions. Users are encouraged to check for updates and read the
latest documentation.
"""

import json
from random import random

from rocketpy._encoders import RocketPyEncoder


class MultivariateRejectionSampler:
    """Class that performs Multivariate Rejection Sampling (MRS) from MonteCarlo
    results.
    """

    def __init__(
        self,
        montecarlo_filepath,
        mrs_filepath,
        distribution_dict,
    ):
        """Initializes Multivariate Rejection Sampler (MRS) class

        Parameters
        ----------
        montecarlo_filepath : str
            Filepath prefixes to the files created from a MonteCarlo simulation
            results.
        mrs_filepath : str
            Filepath prefix to MRS obtained samples. The files created follow the same
            structure as those created by the MonteCarlo class.
        distribution : dict
            Dictionary whose keys contain the name whose distribution changed. The values
            are tuples or lists with two entries. The first entry is a probability
            density (mass) function for the old distribution, while the second entry
            is the probability density function for the new distribution.

        Returns
        -------
        None
        """
        self.montecarlo_filepath = montecarlo_filepath
        self.mrs_filepath = mrs_filepath
        self.distribution_dict = distribution_dict
        self.original_sample_size = 0
        self.sup_ratio = 1
        self.expected_sample_size = None
        self.final_sample_size = None
        # TODO: is there a better way to construct input/output_list?
        # Iterating and appending over lists is costly. However, the
        # alternative, reading the file twice to get the number of lines,
        # also does not seem to be a good option.
        self.output_list = []
        self.input_list = []
        self.__setup_input()
        self.__load_output()

    def __setup_input(self):
        """Loads, validate and compute information from monte carlo
        input with a single read from the file.

        This function does three things:
        1) Load: Loads the input data from MonteCarlo into python
        objects so the sampling process does not require reading from
        disk;
        2) Validate: Validates that the keys in 'distribution_dict' exist in
        the input json created by the monte carlo;
        3) Compute: Computes the supremum of the probability ratios, used in the
        sample function.

        While these three tasks could be disentangled to get clearer
        code, the implementation as done here only requires a single
        read from disk.
        """
        input_filename = f"{self.montecarlo_filepath}.inputs.txt"

        try:
            input_file = open(input_filename, "r+", encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Input file from monte carlo {input_filename} " "not found!"
            ) from e

        for line in input_file.readlines():
            try:
                # loads data
                line_json = json.loads(line)
                self.input_list.append(line_json)
                self.original_sample_size += 1

                prob_ratio = 1
                for parameter in self.distribution_dict.keys():
                    # checks dictionary keys
                    if parameter not in line_json.keys():
                        raise ValueError(
                            f"Parameter key {parameter} from 'distribution_dict' "
                            "not found in input file!"
                        )
                    parameter_value = line_json[parameter]

                    prob_ratio *= self.__compute_probability_ratio(
                        parameter, parameter_value
                    )
                # updates the supremum of the ratio
                self.sup_ratio = max(self.sup_ratio, prob_ratio)
            except Exception as e:
                raise ValueError(
                    "An error occurred while reading "
                    f"the monte carlo input file {input_filename}!"
                ) from e

        self.expected_sample_size = self.original_sample_size // self.sup_ratio
        input_file.close()

    def __load_output(self):
        """Load data from monte carlo outputs."""
        output_filename = f"{self.montecarlo_filepath}.outputs.txt"
        sample_size_output = 0  # sanity check

        try:
            output_file = open(output_filename, "r+", encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Output file from monte carlo {output_filename} " "not found!"
            ) from e

        for line in output_file.readlines():
            try:
                line_json = json.loads(line)
                self.output_list.append(line_json)
                sample_size_output += 1
            except Exception as e:
                raise ValueError(
                    "An error occurred while reading "
                    f"the monte carlo output file {output_filename}!"
                ) from e

        if self.original_sample_size != sample_size_output:
            raise ValueError(
                "Monte carlo input and output files have a different "
                "number of samples!"
            )

        output_file.close()

    def sample(self):
        """Performs rejection sampling and saves data

        Returns
        -------
        None
        """

        mrs_input_file = open(f"{self.mrs_filepath}.inputs.txt", "w+", encoding="utf-8")
        mrs_output_file = open(
            f"{self.mrs_filepath}.outputs.txt", "w+", encoding="utf-8"
        )
        mrs_error_file = open(f"{self.mrs_filepath}.errors.txt", "w+", encoding="utf-8")

        # compute sup ratio
        for line_input_json, line_output_json in zip(self.input_list, self.output_list):
            acceptance_prob = 1 / self.sup_ratio  # probability the sample is accepted
            for parameter in self.distribution_dict.keys():
                parameter_value = line_input_json[parameter]
                acceptance_prob *= self.__compute_probability_ratio(
                    parameter,
                    parameter_value,
                )
            # sample is accepted, write output
            if random() < acceptance_prob:
                mrs_input_file.write(
                    json.dumps(line_input_json, cls=RocketPyEncoder) + "\n"
                )
                mrs_output_file.write(
                    json.dumps(line_output_json, cls=RocketPyEncoder) + "\n"
                )

        mrs_input_file.close()
        mrs_output_file.close()
        mrs_error_file.close()

    def __compute_probability_ratio(self, parameter, parameter_value):
        """Computes the ratio of the new probability to the old probability

        Parameters
        ----------
        parameter : str
            Name of the parameter to evaluate the probability.
        parameter_value : any
            Value of the parameter to be passed to the density functions.

        Returns
        -------
        float
            The ratio of the new probability density function (numerator)
            to the old one (denominator).

        Raises
        ------
        ValueError
            Raises exception if an error occurs when computing the ratio.
        """
        try:
            old_pdf = self.distribution_dict[parameter][0]
            new_pdf = self.distribution_dict[parameter][1]
            probability_ratio = new_pdf(parameter_value) / old_pdf(parameter_value)
        except Exception as e:
            raise ValueError(
                "An error occurred while evaluating the "
                "ratio for 'distribution_dict' probability "
                f"parameter key {parameter}!"
            ) from e

        return probability_ratio
