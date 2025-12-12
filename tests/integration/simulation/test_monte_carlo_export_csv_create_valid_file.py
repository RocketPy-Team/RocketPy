import os
import csv


def test_csv_export(monte_carlo_calisto):
    mc = monte_carlo_calisto
    mc.simulate(3)

    filename = "temp_mc_export.csv"
    """
    tests that results of monte carlo are exported to a CSV file
    """
    mc.export_csv(filename)

    assert os.path.exists(filename)

    with open(filename, newline="") as f:
        reader = list(csv.reader(f))

    # Header exists
    assert len(reader[0]) > 0

    # Should have 3 rows of data after header
    assert len(reader) == 4

    os.remove(filename)
