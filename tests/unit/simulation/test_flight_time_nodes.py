"""Module to test everything related to the TimeNodes class and it's subclass
TimeNode.
"""

# from rocketpy.rocket import Parachute, _Controller


def test_time_nodes_init(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    assert len(time_nodes) == 0


def test_time_nodes_getitem(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    time_nodes.add_node(1.0, [], [], [])
    assert isinstance(time_nodes[0], flight_calisto.TimeNodes.TimeNode)
    assert time_nodes[0].t == 1.0


def test_time_nodes_len(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    assert len(time_nodes) == 0


def test_time_nodes_add(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    example_node = flight_calisto.TimeNodes.TimeNode(1.0, [], [], [])
    time_nodes.add(example_node)
    assert len(time_nodes) == 1
    assert isinstance(time_nodes[0], flight_calisto.TimeNodes.TimeNode)
    assert time_nodes[0].t == 1.0


def test_time_nodes_add_node(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    time_nodes.add_node(2.0, [], [], [])
    assert len(time_nodes) == 1
    assert time_nodes[0].t == 2.0
    assert len(time_nodes[0].parachutes) == 0
    assert len(time_nodes[0].callbacks) == 0


# def test_time_nodes_add_parachutes(
#     flight_calisto, calisto_drogue_chute, calisto_main_chute
# ): # TODO: implement this test


# def test_time_nodes_add_controllers(flight_calisto):
# TODO: implement this test


def test_time_nodes_sort(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    time_nodes.add_node(3.0, [], [], [])
    time_nodes.add_node(1.0, [], [], [])
    time_nodes.add_node(2.0, [], [], [])
    time_nodes.sort()
    assert len(time_nodes) == 3
    assert time_nodes[0].t == 1.0
    assert time_nodes[1].t == 2.0
    assert time_nodes[2].t == 3.0


def test_time_nodes_merge(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    time_nodes.add_node(1.0, [], [], [])
    time_nodes.add_node(1.0, [], [], [])
    time_nodes.add_node(2.0, [], [], [])
    time_nodes.merge()
    assert len(time_nodes) == 2
    assert time_nodes[0].t == 1.0
    assert len(time_nodes[0].parachutes) == 0
    assert len(time_nodes[0].callbacks) == 0
    assert time_nodes[1].t == 2.0
    assert len(time_nodes[1].parachutes) == 0
    assert len(time_nodes[1].callbacks) == 0


def test_time_nodes_flush_after(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    time_nodes.add_node(1.0, [], [], [])
    time_nodes.add_node(2.0, [], [], [])
    time_nodes.add_node(3.0, [], [], [])
    time_nodes.flush_after(1)
    assert len(time_nodes) == 2
    assert time_nodes[0].t == 1.0
    assert time_nodes[1].t == 2.0


def test_time_node_init(flight_calisto):
    node = flight_calisto.TimeNodes.TimeNode(1.0, [], [], [])
    assert node.t == 1.0
    assert len(node.parachutes) == 0
    assert len(node.callbacks) == 0


def test_time_node_lt(flight_calisto):
    node1 = flight_calisto.TimeNodes.TimeNode(1.0, [], [], [])
    node2 = flight_calisto.TimeNodes.TimeNode(2.0, [], [], [])
    assert node1 < node2
    assert not node2 < node1


def test_time_node_repr(flight_calisto):
    node = flight_calisto.TimeNodes.TimeNode(1.0, [], [], [])
    assert isinstance(repr(node), str)


def test_time_nodes_repr(flight_calisto):
    time_nodes = flight_calisto.TimeNodes()
    assert isinstance(repr(time_nodes), str)
