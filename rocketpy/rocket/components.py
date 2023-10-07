from collections import namedtuple


class Components:
    """A Collection Class to hold components of the Rocket class. Each component
    is an object that is added to the rocket at a specific position relative to
    the rocket's coordinate system origin. Components can be added to the rocket
    using the 'add' methods. This class is currently used specifically for
    holding aerodynamic surfaces.

    Attributes
    ----------
    _components : list of namedtuple
        A list of named tuples representing all the components and their
        positions relative to the rocket.
    component_tuple : namedtuple
        A named tuple representing a component and its position within the
        rocket.
    """

    def __init__(self):
        """Initialize an empty components list instance."""
        self.component_tuple = namedtuple("component_tuple", "component position")
        self._components = []

    def __repr__(self):
        """Return a string representation of the Components instance."""
        components_str = "\n".join(
            [
                f"\tComponent: {str(c.component):80} Position: {c.position:>6.3f}"
                for c in self._components
            ]
        )
        return f"Components:\n{components_str}"

    def __len__(self):
        """Return the number of components in the list of components."""
        return len(self._components)

    def __getitem__(self, index):
        """Return the component at the specified index in the list of
        components."""
        return self._components[index]

    def __iter__(self):
        """Return an iterator over the list of components."""
        return iter(self._components)

    def add(self, component, position):
        """Add a component to the list of components.

        Parameters
        ----------
        component : Any
            The component to be added to the rocket.
        position : int, float
            The position of the component relative to the rocket's
            coordinate system origin.

        Returns
        -------
        None
        """
        self._components.append(self.component_tuple(component, position))

    def get_by_type(self, component_type):
        """Search the list of components and return a list with all the
        components of the given type.

        Parameters
        ----------
        component_type : type
            The type of component to be returned.

        Returns
        --------
        list
            A list of components matching the specified type.
        """
        component_type_list = [
            c.component
            for c in self._components
            if isinstance(c.component, component_type)
        ]
        return component_type_list

    def get_tuple_by_type(self, component_type):
        """Search the list of components and return a list with all the components
        of the given type.

        Parameters
        ----------
        component_type : type
            The type of component to be returned.

        Returns
        --------
        list
            A list of components matching the specified type.
        """
        component_type_list = [
            c for c in self._components if isinstance(c.component, component_type)
        ]
        return component_type_list

    def get_positions(self):
        """Return a list of all the positions of the components in the list of
        components.

        Returns
        -------
        list
            A list of all the positions of the components in the list of
            components.
        """
        return [c.position for c in self._components]

    def remove(self, component):
        """Remove a component from the list of components. If more than one
        instance of the same component is present in the list, only the first
        instance is removed.

        Parameters
        ----------
        component : Any
            The component to be removed from the rocket.

        Returns
        --------
        None
        """
        for index, comp in enumerate(self._components):
            if comp.component == component:
                self._components.pop(index)
                break
        else:
            raise Exception(f"Component {component} not found in components {self}")

    def pop(self, index=-1):
        """Pop a component from the list of components.

        Parameters
        ----------
        index : int
            The index of the component to be removed from the list of
            components. If no index is specified, the last component is
            removed.

        Returns
        -------
        component : Any
            The component removed from the list of components.
        """
        return self._components.pop(index)

    def clear(self):
        """Clear all components from the list of components.

        Returns
        -------
        None
        """
        self._components.clear()

    def sort_by_position(self, reverse=False):
        """Sort the list of components by position.

        Parameters
        ----------
        reverse : bool
            If True, sort in descending order. If False, sort in ascending
            order.

        Returns
        -------
        None
        """
        self._components.sort(key=lambda x: x.position, reverse=reverse)
        return None
