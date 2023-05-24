__author__ = "Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from collections import namedtuple


class Components:
    """A group of components, each of which has a position"""

    def __init__(self):
        self.component_position = namedtuple("component_tuple", "component position")
        self._components = []

    def __repr__(self):
        return repr(self._components)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        return self._components[index]

    def __iter__(self):
        return iter(self._components)

    def add(self, component, position):
        """Add a component to the list of components"""
        self._components.append(self.component_position(component, position))

    def get_by_type(self, component_type):
        """Get the component of a specified type"""
        component_type_list = [
            c.component
            for c in self._components
            if isinstance(c.component, component_type)
        ]
        return (
            component_type_list[0]
            if len(component_type_list) == 1
            else component_type_list
        )

    def remove(self, component):
        """Remove a component from the list of components"""
        for index, comp in enumerate(self._components):
            if comp.component == component:
                self._components.pop(index)
                break
        else:
            raise Exception(f"Component {component} not found in components {self}")

    def pop(self, index=-1):
        """Pop a component from the list of components"""
        return self._components.pop(index)

    def clear(self):
        """Clear all components from the list of components"""
        self._components.clear()
