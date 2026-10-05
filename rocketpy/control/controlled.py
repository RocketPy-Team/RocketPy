class Controlled:
    """The objects a controller drives, read by name or by position.

    ``context.controlled`` is one of these inside a controller function. An
    object given a name is read by that name, ``context.controlled.air_brakes``.
    Any object can be read by its position, ``context.controlled[0]``, and
    looping over ``context.controlled`` gives every object in the order they
    were passed to the controller.
    """

    def __init__(self, objects, names=()):
        """Hold the controlled objects.

        Parameters
        ----------
        objects : sequence of object
            The objects the controller drives.
        names : sequence of str, optional
            The name of each object, in the same order. Default is no names.
        """
        self._objects = tuple(objects)
        self._names = tuple(names)
        for name, controlled_object in zip(self._names, self._objects):
            setattr(self, name, controlled_object)

    def __getattr__(self, name):
        # Only reached for a name no object was given.
        if name.startswith("_"):
            raise AttributeError(name)
        available = ", ".join(self._names) or "none; read them by position"
        raise AttributeError(
            f"This controller has no controlled object named {name!r}. The "
            f"names available are: {available}."
        )

    def __getitem__(self, index):
        return self._objects[index]

    def __iter__(self):
        return iter(self._objects)

    def __len__(self):
        return len(self._objects)

    def __repr__(self):
        return f"Controlled(names={self._names!r}, objects={self._objects!r})"
