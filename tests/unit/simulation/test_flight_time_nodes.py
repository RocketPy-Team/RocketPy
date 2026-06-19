"""Module to test everything related to the _TimeNodes class and its subclass
_TimeNode.
"""

from rocketpy.simulation.helpers.flight_phase import _TimeNode, _TimeNodes


def test_time_nodes_init():
    time_nodes = _TimeNodes()
    assert len(time_nodes) == 0


def test_time_nodes_getitem():
    time_nodes = _TimeNodes()
    time_nodes.add_node(1.0, [])
    assert isinstance(time_nodes[0], _TimeNode)
    assert time_nodes[0].t == 1.0


def test_time_nodes_len():
    time_nodes = _TimeNodes()
    assert len(time_nodes) == 0


def test_time_nodes_add():
    time_nodes = _TimeNodes()
    example_node = _TimeNode(1.0, [])
    time_nodes.add(example_node)
    assert len(time_nodes) == 1
    assert isinstance(time_nodes[0], _TimeNode)
    assert time_nodes[0].t == 1.0


def test_time_nodes_add_node():
    time_nodes = _TimeNodes()
    time_nodes.add_node(2.0, [])
    assert len(time_nodes) == 1
    assert time_nodes[0].t == 2.0
    assert len(time_nodes[0].events) == 0


def test_time_nodes_sort():
    time_nodes = _TimeNodes()
    time_nodes.add_node(3.0, [])
    time_nodes.add_node(1.0, [])
    time_nodes.add_node(2.0, [])
    time_nodes.sort()
    assert len(time_nodes) == 3
    assert time_nodes[0].t == 1.0
    assert time_nodes[1].t == 2.0
    assert time_nodes[2].t == 3.0


def test_time_nodes_merge():
    time_nodes = _TimeNodes()
    time_nodes.add_node(1.0, [])
    time_nodes.add_node(1.0, [])
    time_nodes.add_node(2.0, [])
    time_nodes.merge()
    assert len(time_nodes) == 2
    assert time_nodes[0].t == 1.0
    assert len(time_nodes[0].events) == 0
    assert time_nodes[1].t == 2.0
    assert len(time_nodes[1].events) == 0


def test_time_nodes_flush_after():
    time_nodes = _TimeNodes()
    time_nodes.add_node(1.0, [])
    time_nodes.add_node(2.0, [])
    time_nodes.add_node(3.0, [])
    time_nodes.flush_after(1)
    assert len(time_nodes) == 2
    assert time_nodes[0].t == 1.0
    assert time_nodes[1].t == 2.0


def test_time_node_init():
    node = _TimeNode(1.0, [])
    assert node.t == 1.0
    assert len(node.events) == 0


def test_time_node_lt():
    node1 = _TimeNode(1.0, [])
    node2 = _TimeNode(2.0, [])
    assert node1 < node2
    assert (node2 < node1) is False


def test_time_node_repr():
    node = _TimeNode(1.0, [])
    assert isinstance(repr(node), str)


def test_time_nodes_repr():
    time_nodes = _TimeNodes()
    assert isinstance(repr(time_nodes), str)
