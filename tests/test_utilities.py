from rocketpy import utilities


def test_compute_CdS_from_drop_test():
    assert (
        utilities.compute_CdS_from_drop_test(31.064, 18, 1.0476) == 0.3492311157844522
    )
