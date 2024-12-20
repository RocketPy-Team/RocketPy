"""
Multivariate Rejection Sampling Module for RocketPy

Notes
-----
This module is still under active development, and some features or attributes may
change in future versions. Users are encouraged to check for updates and read the
latest documentation.
"""

import json
from dataclasses import dataclass
from random import random

from rocketpy._encoders import RocketPyEncoder


@dataclass
class SampleInformation:
    """Sample information used in the MRS"""

    inputs_json: dict = None
    outputs_json: dict = None
    probability_ratio: float = None
    acceptance_probability: float = None


class MultivariateRejectionSampler:
    """Class that performs Multivariate Rejection Sampling (MRS) from MonteCarlo
    results. The class currently assumes that all input variables are independent
    when performing the

    Attributes
    ----------
    """

    def __init__(
        self,
        monte_carlo_filepath,
        mrs_filepath,
    ):
        """Initializes Multivariate Rejection Sampler (MRS) class

        Parameters
        ----------
        monte_carlo_filepath : str
            Filepath prefixes to the files created from a MonteCarlo simulation
            results and used as input for resampling.
        mrs_filepath : str
            Filepath prefix to MRS obtained samples. The files created follow the same
            structure as those created by the MonteCarlo class but now containing the
            selected sub-samples.

        Returns
        -------
        None
        """
        self.monte_carlo_filepath = monte_carlo_filepath
        self.mrs_filepath = mrs_filepath
        self.distribution_dict = None
        self.original_sample_size = 0
        self.sup_ratio = 1
        self.expected_sample_size = None
        self.final_sample_size = None
        self.input_variables_names = set()
        self.output_variables_names = set()
        self.all_sample_list = []
        self.accepted_sample_list = []
        self.__setup_input()
        self.__load_output()

    def __setup_input(self):
        """Loads input information from monte carlo in a SampleInformation
        object
        """
        input_filename = f"{self.monte_carlo_filepath}.inputs.txt"

        try:
            input_file = open(input_filename, "r+", encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Input file from monte carlo {input_filename} not found!"
            ) from e

        try:
            for line in input_file.readlines():
                sample_info = SampleInformation()
                line_json = json.loads(line)
                sample_info.inputs_json = line_json
                self.all_sample_list.append(sample_info)

                # sets and validates input variables names
                if not self.input_variables_names:
                    self.input_variables_names = set(line_json.keys())
                else:
                    if self.input_variables_names != set(line_json.keys()):
                        raise ValueError(
                            "Input file from Monte Carlo contains different "
                            "variables for different lines"
                        )
                self.original_sample_size += 1
        except Exception as e:
            raise ValueError(
                "An error occurred while reading "
                f"the monte carlo input file {input_filename}!"
            ) from e

        input_file.close()

    def __load_output(self):
        """Loads output information from monte carlo in a SampleInformation
        object.
        """
        output_filename = f"{self.monte_carlo_filepath}.outputs.txt"
        sample_size_output = 0  # sanity check

        try:
            output_file = open(output_filename, "r+", encoding="utf-8")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Output file from monte carlo {output_filename} " "not found!"
            ) from e

        try:
            for line in output_file.readlines():
                if sample_size_output > self.original_sample_size:
                    raise ValueError(
                        "Monte carlo output has more lines than the input file!"
                    )
                line_json = json.loads(line)
                self.all_sample_list[sample_size_output].outputs_json = line_json
                sample_size_output += 1
        except Exception as e:
            raise ValueError(
                "An error occurred while reading "
                f"the monte carlo output file {output_filename}!"
            ) from e
        if self.original_sample_size > sample_size_output:
            raise ValueError(
                "Monte carlo output file has fewer lines than the input file!"
            )

        output_file.close()

    def __validate_distribution_dict(self, distribution_dict):
        """Checks that the variables passed in the distribution dictionary were
        in the input file.

        """
        input_variables_names = set(distribution_dict.keys())
        for variable in input_variables_names:
            if variable not in self.input_variables_names:
                raise ValueError(
                    f"Variable key {variable} from 'distribution_dict' "
                    "not found in input file!"
                )

    def sample(self, distribution_dict):
        """Performs rejection sampling and saves data

        Parameters
        ----------
        distribution : dict
            Dictionary whose keys contain the name whose distribution changed.
            The values are tuples or lists with two entries. The first entry is
            a probability density (mass) function for the old distribution,
            while the second entry is the probability density function for the
            new distribution.

        Returns
        -------
        None
        """

        self.__validate_distribution_dict(distribution_dict)

        mrs_input_file = open(f"{self.mrs_filepath}.inputs.txt", "w+", encoding="utf-8")
        mrs_output_file = open(
            f"{self.mrs_filepath}.outputs.txt", "w+", encoding="utf-8"
        )
        mrs_error_file = open(f"{self.mrs_filepath}.errors.txt", "w+", encoding="utf-8")

        self.__setup_probabilities(distribution_dict)

        try:
            for sample in self.all_sample_list:
                if random() < sample.acceptance_probability:
                    mrs_input_file.write(
                        json.dumps(sample.inputs_json, cls=RocketPyEncoder) + "\n"
                    )
                    mrs_output_file.write(
                        json.dumps(sample.outputs_json, cls=RocketPyEncoder) + "\n"
                    )
        except Exception as e:
            raise ValueError(
                "An error occurred while writing the selected sample to the "
                "output files"
            ) from e

        mrs_input_file.close()
        mrs_output_file.close()
        mrs_error_file.close()

    def __setup_probabilities(self, distribution_dict):
        """Computes the probability ratio, probability ratio supremum and acceptance
        probability for each sample.

        Parameters
        ----------
        distribution : dict
            Dictionary whose keys contain the name whose distribution changed. The values
            are tuples or lists with two entries. The first entry is a probability
            density (mass) function for the old distribution, while the second entry
            is the probability density function for the new distribution.
        """
        self.sup_ratio = 1
        for sample in self.all_sample_list:
            sample.probability_ratio = self.__compute_probability_ratio(
                sample, distribution_dict
            )
            self.sup_ratio = max(self.sup_ratio, sample.probability_ratio)

        for sample in self.all_sample_list:
            sample.acceptance_probability = sample.probability_ratio / self.sup_ratio
        self.expected_sample_size = self.original_sample_size // self.sup_ratio

    def __compute_probability_ratio(self, sample, distribution_dict):
        """Computes the ratio of the new probability to the old probability
        for the given sample

        Parameters
        ----------
        sample: SampleInformation
            Sample information used to extract the values to evaluate the
            distributions pdf.
        distribution : dict
            Dictionary whose keys contain the name whose distribution changed. The values
            are tuples or lists with two entries. The first entry is a probability
            density (mass) function for the old distribution, while the second entry
            is the probability density function for the new distribution.

        Raises
        ------
        ValueError
            Raises exception if an error occurs when computing the ratio.
        """
        probability_ratio = 1
        try:
            for variable in distribution_dict.keys():
                value = sample.inputs_json[variable]
                old_pdf = distribution_dict[variable][0]
                new_pdf = distribution_dict[variable][1]
                probability_ratio *= new_pdf(value) / old_pdf(value)
        except Exception as e:
            raise ValueError(
                "An error occurred while evaluating the probability ratio"
            ) from e

        return probability_ratio
