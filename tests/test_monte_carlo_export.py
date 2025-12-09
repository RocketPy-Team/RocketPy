import os
import json

def test_json_export(monte_carlo_calisto):
    mc = monte_carlo_calisto
    mc.simulate(3)

    filename = "temp_test_output.json"
    mc.export_json(filename)

    assert os.path.exists(filename)

    with open(filename, "r") as f:
        data = json.load(f)

    # Assert dictionary keys exist
    assert len(data.keys()) > 0

    # Check that at least one key corresponds to a list of simulation results
    list_keys = [k for k, v in data.items() if isinstance(v, list)]

    # There must be at least 1 Monte Carlo-dependent field
    assert len(list_keys) > 0

    first_list_key = list_keys[0]
    assert len(data[first_list_key]) == 3

    os.remove(filename)
