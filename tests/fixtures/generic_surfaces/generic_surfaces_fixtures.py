import pandas as pd
import pytest


@pytest.fixture(scope="session")
def filename_valid_coeff(tmpdir_factory):
    """Creates temporary files used to test if generic surfaces
    initializes correctly from CSV files"""
    filename = tmpdir_factory.mktemp("aero_surface_data").join("valid_coefficients.csv")
    pd.DataFrame(
        {
            "alpha": [0, 1, 2, 3, 0.1],
            "mach": [3, 2, 1, 0, 0.2],
            "cL": [4, 2, 2, 4, 5],
        }
    ).to_csv(filename, index=False)

    return filename


@pytest.fixture(
    params=(
        {
            "alpha": [0, 1, 2, 3, 0.1],
            "cL": [4, 2, 2, 4, 5],
            "mach": [3, 2, 1, 0, 0.2],
        },
        {
            "a": [0, 1, 2, 3, 0.1],
            "b": [4, 2, 2, 4, 5],
        },
    )
)
def filename_invalid_coeff(tmpdir_factory, request):
    """Creates temporary CSV files used to test if generic surfaces
    raises errors when initialized incorrectly from CSV files"""
    filename = tmpdir_factory.mktemp("aero_surface_data").join(
        "tmp_invalid_coefficients.csv"
    )
    pd.DataFrame(request.param).to_csv(filename, index=False)

    return filename
