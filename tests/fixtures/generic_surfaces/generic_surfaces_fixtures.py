import pandas as pd
import pytest


@pytest.fixture(scope="session")
def filename_valid_coeff(tmpdir_factory):
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
        [0, 1, 2, 3],
    )
)
def filename_invalid_coeff(tmpdir_factory, request):
    filename = tmpdir_factory.mktemp("aero_surface_data").join(
        "tmp_invalid_coefficients.csv"
    )
    if isinstance(request.param, dict):
        pd.DataFrame(request.param).to_csv(filename, index=False)
    else:
        pd.DataFrame(request.param).to_csv(filename, index=False, header=False)

    return filename
