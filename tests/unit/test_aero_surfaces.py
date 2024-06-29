import pytest
from rocketpy import NoseCone

NOSECONE_LENGTH = 1
NOSECONE_BASE_RADIUS = 1
NOSECONE_ROCKET_RADIUS = 1
NOSECONE_KIND = "powerseries"


@pytest.mark.parametrize(
    "power, bluffness",
    [
        (0, None),
        (None, None),
        (0, 0.5),
        (None, 0.5),
        (0.5, 0.1),
        (-1, None),
        (10, None),
    ],
)
def test_powerseries_nosecones_invalid_inputs(power, bluffness):
    """Checks if Exceptions are raised correctly when invalid inputs
    are passed for the creation of power series nose cones
    """

    # Tests for invalid power parameter
    with pytest.raises(ValueError):
        NoseCone(
            length=NOSECONE_LENGTH,
            base_radius=NOSECONE_BASE_RADIUS,
            rocket_radius=NOSECONE_ROCKET_RADIUS,
            kind=NOSECONE_KIND,
            power=power,
            bluffness=bluffness,
        )


@pytest.mark.parametrize(
    "power, invalid_power, new_power",
    [
        (0.1, -1, 0.05),
        (0.5, 0, 0.7),
        (0.9, 10, 1),
    ],
)
def test_powerseries_nosecones_setters(power, invalid_power, new_power):
    """Checks if Exceptions are raised correctly when the 'power' or
    'bluffness' attributes are changed to invalid values. Also checks
    that modifying the 'power' attribute also modifies the 'k' attribute.
    """
    test_nosecone = NoseCone(
        length=NOSECONE_LENGTH,
        base_radius=NOSECONE_BASE_RADIUS,
        rocket_radius=NOSECONE_ROCKET_RADIUS,
        kind=NOSECONE_KIND,
        power=power,
        bluffness=None,
    )
    # Test invalid power value modification
    with pytest.raises(ValueError):
        test_nosecone.power = invalid_power

    # Test invalid bluffness value modification
    with pytest.raises(ValueError):
        test_nosecone.bluffness = 0.5

    # Checks if self.k was updated correctly
    test_nosecone.power = new_power
    expected_k = (2 * new_power) / ((2 * new_power) + 1)

    assert pytest.approx(test_nosecone.k) == expected_k
