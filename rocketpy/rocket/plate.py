import math as m

import numpy as np
from matplotlib.path import Path

from rocketpy.exceptions import InvalidParameterError
from rocketpy.mathutils import Matrix, Vector
from rocketpy.plots.plate_plots import _PlatePlots
from rocketpy.prints.plate_prints import _PlatePrints


class Plate:
    """Defines a surface on the rocket fuselage or interior to account for
    soft-iron magnetic distortion affecting magnetometer readings.

    Attributes
    ----------
    Plate.shape : str
        Shape of the plate ("circular", "rectangular", or "personalized").
    Plate.dimensions : float, int, tuple or list
        Dimensions of the plate. For "circular", a float/int representing the
        radius; for "rectangular", a tuple (width, height); for
        "personalized", a list of 3D vertices [x, y, z].
    Plate.material : str
        Material composing the plate ("iron", "carbon_steel", or "personalized").
    Plate.absolute_magnetic_permeability : float
        Absolute magnetic permeability of the ``material`` in H/m (or N/A^2).
    Plate.relative_magnetic_permeability : float
        Ratio of absolute magnetic permeability to the magnetic permeability of
        free space (dimensionless).
    Plate.thickness : float, int
        Thickness of the plate in meters.
    Plate.area : float
        Surface area of the plate in square meters.
    Plate.volume : float
        Volume of the plate in cubic meters.
    Plate._magnetic_distortion_matrices : dict
        Dictionary of soft-iron distortion matrices induced by the plate. Keys
        are position vector tuples (x, y, z) in the Body Axis Coordinate System,
        and values are the corresponding 3x3 Matrix instances.
    Plate.points : list[list]
        Discretized points defining the plate surface in the Body Axis
        Coordinate System.
    Plate.grid_spacing : float, optional
        Spacing between grid points in meters for "personalized" plates.
    Plate.z_points : int, optional
        Number of points along the longitudinal axis for "circular" or
        "rectangular" plates.
    Plate.angular_points : int, optional
        Number of angular points for "circular" or "rectangular" plates.
    Plate.name : str
        Name of the plate.
    """

    def __init__(
        self,
        shape: str,
        dimensions: int | float | list | tuple,
        material: str,
        thickness: int | float,
        absolute_magnetic_permeability: int | float | None = None,
        relative_magnetic_permeability: int | float | None = None,
        grid_spacing: int | float = 0.001,
        z_points: int = 40,
        angular_points: int = 70,
        name: str = "Plate",
    ) -> None:
        """Initializes the Plate.

        Parameters
        ----------
        shape : str
            Shape of the plate. Options are:

            - If "circular": the plate will be built as a circle, and the parameter
              ``dimensions`` refers to the radius in meters.
            - If "rectangular": the plate will be built a rectangle, and the parameter
              ``dimensions`` refers to the sides length in meters.
            - If "personalized": the plate is created by the
              vertices defined in ``dimensions``.
        dimensions : float, int, list, tuple
            Dimensions defining the plate geometry depending on ``shape``:

            - If "circular": float or int representing radius in meters.
            - If "rectangular": sequence of [width, height] in meters, or a
              single float/int for a square.
            - If "personalized": sequence of at least 3 non-collinear 3D vertices
              [x, y, z] defined in the User-defined Coordinate System.
        material : str
            Material composing the plate. Allowed values are "iron",
            "carbon_steel", or "personalized".
        thickness : float, int
            Thickness of the plate in meters.
        absolute_magnetic_permeability : float, int, optional
            Absolute magnetic permeability of the ``material`` in H/m. Default is
            None.
        relative_magnetic_permeability : float, int, optional
            Dimensionless ratio of material permeability to free space
            permeability. If defined, it overrides
            ``absolute_magnetic_permeability``. Default is None.
        grid_spacing : float, optional
            Distance between points that form the plate in meters when ``shape`` is
            "personalized". Default is 0.001.
        z_points : int, optional
            Number of points along the longitudinal axis for "circular" or
            "rectangular" plates. Default is 40.
        angular_points : int, optional
            Number of angular points for "circular" or
            "rectangular" plates. Default is 70.
        name : str, optional
            Name of the plate. Default is "Plate".
        """
        self._magnetic_distortion_matrices = {}
        self.points = []
        self.plots = _PlatePlots(self)
        self.prints = _PlatePrints(self)
        self._validate_parameters(
            material,
            absolute_magnetic_permeability,
            relative_magnetic_permeability,
            shape,
            dimensions,
            z_points,
            angular_points,
            grid_spacing,
        )
        self.thickness = thickness
        self.name = name

    def _validate_parameters(
        self,
        material,
        absolute_magnetic_permeability,
        relative_magnetic_permeability,
        shape,
        dimensions,
        z_points,
        angular_points,
        grid_spacing,
    ) -> None:
        """Validates input parameters and defines attributes."""
        self._validate_material(
            material, absolute_magnetic_permeability, relative_magnetic_permeability
        )
        self._validate_shape(shape, dimensions, z_points, angular_points, grid_spacing)

    def _validate_material(
        self, material, absolute_magnetic_permeability, relative_magnetic_permeability
    ) -> None:
        """Validates and sets material magnetic permeability parameters."""
        if not isinstance(material, str):
            raise InvalidParameterError("'material' argument must be a string.")

        mu_0 = 4 * np.pi * 1e-7
        predefined_materials = {"iron": 1.25e-3, "carbon_steel": 1.2e-4}

        if material in predefined_materials:
            self.material = material
            self.absolute_magnetic_permeability = predefined_materials[material]
            self.relative_magnetic_permeability = (
                self.absolute_magnetic_permeability / mu_0
            )

        elif material == "personalized":
            if not isinstance(absolute_magnetic_permeability, (float, int, type(None))):
                raise InvalidParameterError(
                    "The 'absolute_magnetic_permeability' must be None, float, or int."
                )
            if (
                absolute_magnetic_permeability is None
                and relative_magnetic_permeability is None
            ):
                raise InvalidParameterError(
                    "Either 'absolute_magnetic_permeability' or 'relative_magnetic_permeability' must be specified when 'material' is 'personalized'."
                )

            self.material = "personalized"

            if relative_magnetic_permeability is None:
                self.absolute_magnetic_permeability = absolute_magnetic_permeability
                self.relative_magnetic_permeability = (
                    absolute_magnetic_permeability / mu_0
                )

            elif isinstance(relative_magnetic_permeability, (float, int)):
                self.relative_magnetic_permeability = relative_magnetic_permeability
                self.absolute_magnetic_permeability = (
                    relative_magnetic_permeability * mu_0
                )

            else:
                raise InvalidParameterError(
                    "The 'relative_magnetic_permeability' must be None, float, or int."
                )
        else:
            raise InvalidParameterError(
                "'material' argument must be 'iron', 'carbon_steel', or 'personalized'."
            )

    def _validate_shape(
        self, shape, dimensions, z_points, angular_points, grid_spacing
    ) -> None:
        """Validates ``shape`` parameters and assigns attributes."""
        if isinstance(shape, str):
            if shape in ("circular", "rectangular"):
                self._validate_not_personalized_shape(
                    shape, dimensions, z_points, angular_points
                )
            elif shape == "personalized":
                self.shape = shape
                self.dimensions = dimensions
                self.grid_spacing = grid_spacing
                self.z_points = None
                self.angular_points = None
            else:
                raise InvalidParameterError(
                    "'shape' must be 'circular', 'rectangular', or 'personalized'."
                )
        else:
            raise InvalidParameterError("'shape' must be defined as a string.")

    def _validate_not_personalized_shape(
        self, shape, dimensions, z_points, angular_points
    ):
        """Validates parameters for standard geometric shapes ('circular' or 'rectangular')."""
        if shape == "circular":
            if not isinstance(dimensions, (float, int)):
                raise InvalidParameterError(
                    "For 'circular' shape, dimensions must be a float or int representing radius in meters."
                )
            self.shape = "circular"
            self.dimensions = dimensions
            self.z_points = z_points
            self.angular_points = angular_points
            self.grid_spacing = None

        elif shape == "rectangular":
            self.shape = "rectangular"
            self.z_points = z_points
            self.angular_points = angular_points
            self.grid_spacing = None

            if isinstance(dimensions, (float, int)):
                width = float(dimensions)
                height = float(dimensions)
            elif isinstance(dimensions, (list, tuple)) and len(dimensions) == 2:
                width = float(dimensions[0])
                height = float(dimensions[1])
            else:
                raise InvalidParameterError(
                    "For 'rectangular' shape, dimensions must be a float/int (square) or a 2-element sequence [width, height]."
                )
            self.dimensions = (width, height)

    def define_plate_position(
        self,
        rocket,
        position: float | int | None = None,
        height: float | int | None = None,
    ) -> None:
        """Defines the geometry, area, and volume of the plate when attached
        to a rocket instance.

        Parameters
        ----------
        rocket : Rocket
            Rocket instance to which the plate is attached.
        position : float, int, optional
            Angle between the User-defined Coordinate System y-axis and the
            geometric center of the plate in degrees. The angle is positive from
            y to -x. See `Rocket Axes Definition <https://docs.rocketpy.org/en/latest/user/rocket/rocket_axes.html>`_
            for more information.
        height : float, int, optional
            Coordinate of the geometric center along the rocket
            longitudinal axis in the User-defined Coordinate System in
            meters.
        """
        self.generate_points(rocket, position, height)
        if self.shape == "rectangular":
            self.area = self.dimensions[0] * self.dimensions[1]
        elif self.shape == "circular":
            self.area = np.pi * (self.dimensions**2)
        else:  # personalized
            self.area = len(self.points) * (self.grid_spacing**2)
        self.volume = self.area * self.thickness

    def generate_points(
        self,
        rocket,
        position: float | int | None = None,
        height: float | int | None = None,
    ) -> None:
        """Generates the points that represent the plate surface in the Body Axis
        Coordinate System.

        Parameters
        ----------
        rocket : Rocket
            Rocket instance to which the plate is attached.
        position : float, int, optional
            Angle between the User-defined Coordinate System y-axis and the
            geometric center of the plate in degrees. The angle is positive from
            y to -x. See `Rocket Axes Definition <https://docs.rocketpy.org/en/latest/user/rocket/rocket_axes.html>`_
            for more information.
        height : float, int, optional
            Axial coordinate of the geometric center along the rocket
            longitudinal axis in the User-defined Coordinate System in
            meters. Required when the ``shape`` of the plate is "circular" or
            "rectangular".
        """
        self.points = []

        if self.shape == "personalized":
            self._generate_personalized_internal_plate(rocket)
        else:
            if self.shape == "circular":
                upper_z = height + self.dimensions
                lower_z = height - self.dimensions
            else:
                upper_z = height + self.dimensions[1] / 2
                lower_z = height - self.dimensions[1] / 2
            geometric_center_angle = position * (np.pi / 180)
            for z in np.linspace(lower_z, upper_z, self.z_points):
                if not rocket.z_bounds_check(z, frame="ucs")[0]:
                    continue

                r = rocket.general_radius(z, frame="ucs")
                if r <= 1e-6:
                    continue
                alpha, beta = self._define_angles(z, geometric_center_angle, r, height)
                for theta in np.linspace(alpha, beta, self.angular_points):
                    x = -r * np.sin(theta)
                    y = r * np.cos(theta)

                    # Transform to Body Axis Coordinate System (BACS)
                    z_bacs = (z - self.center_of_dry_mass_position) * self._csys
                    if self._csys == -1:  # nose_to_tail
                        self.points.append([-x, y, z_bacs])
                    else:  # tail_to_nose
                        self.points.append([x, y, z_bacs])

    def _define_angles(
        self, z, geometric_center_angle, r, height
    ) -> tuple[float, float]:
        """Calculates the angular span for the definition of the points
        that form the plate at axial position z.

        Parameters
        ----------
        z : float
            Longitudinal coordinate along the rocket axis in meters.
        geometric_center_angle : float
            Angle of the geometric center of the plate in radians.
        r : float
            Rocket fuselage radius at coordinate z in meters.
        height : float
            Longitudinal position of the geometric center of the plate in
            meters.

        Returns
        -------
        alpha : float
            Starting angular bound in radians.
        beta : float
            Ending angular bound in radians.
        """
        if self.shape == "circular":
            alpha, beta = self._circular_angle_calculation(
                z=z, center_z=height, geometric_center_angle=geometric_center_angle, r=r
            )
        else:  # rectangular
            alpha, beta = self._rectangular_angle_calculation(geometric_center_angle, r)
        return alpha, beta

    def _circular_angle_calculation(
        self, z, center_z, geometric_center_angle, r
    ) -> tuple[float, float]:
        """Calculates the angular span for the definition of the points
        that form the plate when it is "circular" at axial position z.

        Parameters
        ----------
        z : float
            Current longitudinal coordinate in meters.
        center_z : float
            Longitudinal coordinate of the geometric center in meters.
        geometric_center_angle : float
            Angular position of the geometric center in radians.
        r : float
            Rocket radius at coordinate z in meters.

        Returns
        -------
        alpha : float
            Starting angular bound in radians.
        beta : float
            Ending angular bound in radians.
        """
        dz = z - center_z
        inside_sqrt = max(self.dimensions**2 - dz**2, 0)
        half_extension_angle = m.sqrt(inside_sqrt) / r
        alpha = geometric_center_angle - half_extension_angle
        beta = geometric_center_angle + half_extension_angle
        return alpha, beta

    def _rectangular_angle_calculation(
        self, geometric_center_angle, r
    ) -> tuple[float, float]:
        """Calculates the angular span for the definition of the points
        that form the plate when it is "rectangular" at axial position z.

        Parameters
        ----------
        geometric_center_angle : float
            Angular position of the geometric center in radians.
        r : float
            Rocket radius at the current coordinate in meters.

        Returns
        -------
        alpha : float
            Starting angular bound in radians.
        beta : float
            Ending angular bound in radians.
        """
        extension_angle = self.dimensions[0] / r
        alpha = geometric_center_angle - extension_angle / 2
        beta = geometric_center_angle + extension_angle / 2
        return alpha, beta

    def _generate_personalized_internal_plate(
        self,
        rocket,
    ) -> None:
        """Generates a planar 2D grid of 3D points bounded by arbitrary vertices,
        constrained within the rocket fuselage.

        Parameters
        ----------
        rocket : Rocket
            Rocket instance to which the plate belongs.
        """
        vertices = self._vertices_definition(rocket)

        centroid = [sum(col) / len(vertices) for col in zip(*vertices)]
        cx, cy, cz = centroid[0], centroid[1], centroid[2]

        ux, uy, uz, vx, vy, vz = self._calculate_uv_frame(vertices)

        uv_vertices = [
            (
                (pt[0] - cx) * ux + (pt[1] - cy) * uy + (pt[2] - cz) * uz,
                (pt[0] - cx) * vx + (pt[1] - cy) * vy + (pt[2] - cz) * vz,
            )
            for pt in vertices
        ]

        u_coords, v_coords = zip(*uv_vertices)
        plate_path = Path(uv_vertices)
        final_3d_points = []

        curr_u = min(u_coords)
        while curr_u <= max(u_coords):
            curr_v = min(v_coords)
            while curr_v <= max(v_coords):
                if plate_path.contains_point((curr_u, curr_v)):
                    p3d_x = cx + (curr_u * ux) + (curr_v * vx)
                    p3d_y = cy + (curr_u * uy) + (curr_v * vy)
                    p3d_z = cz + (curr_u * uz) + (curr_v * vz)

                    if m.hypot(p3d_x, p3d_y) < rocket.general_radius(
                        p3d_z, frame="bacs"
                    ):
                        final_3d_points.append([p3d_x, p3d_y, p3d_z])

                curr_v += self.grid_spacing
            curr_u += self.grid_spacing

        self.points = final_3d_points

    def _vertices_definition(self, rocket) -> list:
        """Transforms vertices in user-defined coordinate system into
        the Body Axis Coordinate System.

        Parameters
        ----------
        rocket : Rocket
            Rocket instance to which the plate belongs.

        Returns
        -------
        vertices : list
            List of 3D vertices transformed into the BACS frame.
        """
        vertices = []
        cdm_user_frame = Vector([0, 0, self.center_of_dry_mass_position])
        if len(self.dimensions) < 3:
            raise InvalidParameterError(
                "At least 3 vertices are required for a personalized plate."
            )
        for pt in self.dimensions:
            self._check_entry_dimensions(pt, rocket)

            # Transform to BACS
            sensor_vec = Vector(pt) - cdm_user_frame
            if self._csys == -1:
                vertices.append([-sensor_vec[0], sensor_vec[1], -sensor_vec[2]])
            else:
                vertices.append([sensor_vec[0], sensor_vec[1], sensor_vec[2]])
        collinear = []
        for i in range(len(vertices) - 2):
            e1 = Vector(vertices[i + 1]) - Vector(vertices[i])
            e2 = Vector(vertices[i + 2]) - Vector(vertices[i + 1])
            collinear.append(abs(e1 ^ e2) <= 1e-12)

        if all(collinear):
            raise InvalidParameterError(
                "All vertices of the personalized plate are collinear."
            )
        return vertices

    def _check_entry_dimensions(self, pt, rocket) -> None:
        """Validates that a vertex lies within the rocket's longitudinal and radial boundaries.

        Parameters
        ----------
        pt : list, tuple, Vector
            Coordinates [x, y, z] in the User-defined Coordinate System.
        rocket : Rocket
            Rocket instance in which boundaries are checked.
        """
        if not rocket.z_bounds_check(pt[2], frame="ucs")[0]:
            raise InvalidParameterError(
                f"The z coordinate {pt[2]} of point {pt} is outside the rocket longitudinal bounds."
            )

        if m.hypot(pt[0], pt[1]) > rocket.general_radius(pt[2], frame="ucs"):
            raise InvalidParameterError(
                f"Point {pt} exceeds the rocket fuselage radius at z = {pt[2]} m."
            )

    def _calculate_uv_frame(
        self, vertices
    ) -> tuple[float, float, float, float, float, float]:
        """Calculates an orthonormal planar basis (u, v) from polygon vertices.

        Parameters
        ----------
        vertices : list
            List of 3D vertices defining the planar plate.

        Returns
        -------
        ux : float
            X component of the first in-plane basis unit vector.
        uy : float
            Y component of the first in-plane basis unit vector.
        uz : float
            Z component of the first in-plane basis unit vector.
        vx : float
            X component of the second in-plane basis unit vector.
        vy : float
            Y component of the second in-plane basis unit vector.
        vz : float
            Z component of the second in-plane basis unit vector.
        """
        v0, v1, v2 = vertices[:3]
        nx = (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1])
        ny = (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2])
        nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])

        n_norm = m.hypot(nx, ny, nz)
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

        arb_x, arb_y, arb_z = (
            (0.0, 0.0, 1.0) if (abs(nx) > 0.5 or abs(ny) > 0.5) else (1.0, 0.0, 0.0)
        )

        ux = ny * arb_z - nz * arb_y
        uy = nz * arb_x - nx * arb_z
        uz = nx * arb_y - ny * arb_x
        u_norm = m.hypot(ux, uy, uz)
        if u_norm < 1e-12:
            raise InvalidParameterError("Invalid plate orientation or geometry.")
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm

        vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux

        return ux, uy, uz, vx, vy, vz

    def calculate_soft_iron_distortion_matrix(
        self, position_vector: Vector | list | tuple, frame: str = "ucs"
    ) -> None:
        """Calculates and stores the 3x3 soft-iron magnetic distortion matrix
        induced by the plate at the position indicated by ``position_vector``.

        Parameters
        ----------
        position_vector : Vector, list, tuple
            Coordinates [x, y, z] in meters in the Body Axis Coordinate
            System where the distortion matrix is evaluated.
        """
        if isinstance(self.points, list):
            if self.points:
                induced_matrix = Matrix.zeros()
                diff_magnetic = self.relative_magnetic_permeability - 1.0
                num_points = len(self.points)
                dv = self.volume / num_points
                dipole_scalar = (diff_magnetic * dv) / (4.0 * np.pi)
                position_vector = self._position_vector_to_bacs(position_vector, frame)
                for point in self.points:
                    r_v = position_vector - Vector(point)
                    r = abs(r_v)
                    r_unit = r_v / r

                    rx, ry, rz = r_unit[0], r_unit[1], r_unit[2]

                    projection_tensor = Matrix(
                        [
                            [rx * rx, rx * ry, rx * rz],
                            [ry * rx, ry * ry, ry * rz],
                            [rz * rx, rz * ry, rz * rz],
                        ]
                    )

                    dipole_kernel = (3.0 * projection_tensor - Matrix.identity()) / (
                        r**3
                    )

                    induced_matrix = induced_matrix + (dipole_scalar * dipole_kernel)

                self._magnetic_distortion_matrices[tuple(position_vector)] = (
                    induced_matrix
                )
            else:
                raise InvalidParameterError(
                    "The plate points list is empty. Add the plate to a rocket before calculating the distortion matrix."
                )
        else:
            raise InvalidParameterError("The points attribute must be a list.")

    def _position_vector_to_bacs(
        self,
        position_vector: list | tuple | Vector,
        frame: str = "bacs",
    ) -> Vector:
        """Transforms, if necessary, the position_vector to the BACS frame.

        Parameters
        ----------
        position_vector : list, tuple, Vector
            Coordinates [x, y, z] in meters in the specified ``frame``
            in which we want to calculate the soft iron distortion.
        frame : str, optional
            Frame in which the ``position_vector`` is given. It can either be "bacs"
            (body axis coordinate system) or "ucs" (user defined coordinate
            system). Default is "ucs".

        Returns
        -------
        position_vector_bacs : Vector
            Vector coordinates [x, y, z] in meters in the BACS frame.
        """
        if not isinstance(frame, str):
            raise InvalidParameterError("'frame' must be a string.")

        frame_lower = frame.lower()
        if frame_lower not in ("ucs", "bacs"):
            raise InvalidParameterError(
                f"Invalid frame '{frame}'. Must be 'ucs' or 'bacs'."
            )
        if isinstance(position_vector, (list, tuple, Vector)):
            pos_v = Vector(position_vector)
        else:
            raise InvalidParameterError(
                "'position_vector' must be a tuple, list, or Vector instance."
            )
        if frame_lower == "bacs":
            position_vector_bacs = pos_v
        else:
            cdm_user_frame = Vector([0, 0, self.center_of_dry_mass_position])
            delta = pos_v - cdm_user_frame
            if self._csys == -1:  # nose to tail
                position_vector_bacs = Vector([-delta[0], delta[1], -delta[2]])
            else:  # tail to nose
                position_vector_bacs = delta

        return position_vector_bacs

    def draw_3d(
        self,
        color: str = "teal",
        marker: str = "h",
        elev: float | int | None = None,
        azim: float | int | None = None,
        filename: str | None = None,
    ) -> None:
        """Plots the 3D scatter plot of the points forming the plate
        that are used to model soft-iron magnetic distortion.

        Parameters
        ----------
        color : str, optional
            Color of the scatter points. A full list of color names can be found
            at: https://matplotlib.org/stable/gallery/color/named_colors
            Default is "teal".
        marker : str, optional
            Shape of the markers representing the points. A full
            list of markers can be found at:
            https://matplotlib.org/stable/api/markers_api.html
            Default is "h".
        elev : float, int, optional
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. If None, the default view is used.
            Default is None.
        azim : float, int, optional
            The azimuthal angle in degrees rotates the camera about the vertical
            axis. If None, the default view is used. Default is None.
        filename : str, optional
            The path the plot should be saved to. If None, the plot will be shown instead
            of saved. Supported file formats include: eps, jpg, jpeg, pdf, pgf, png, ps,
            raw, rgba, svg, svgz, tif, tiff, and webp. Default is None.
        """
        self.plots.draw_3d(color, marker, elev, azim, filename)

    def _rocket_belonging(self, rocket) -> None:
        """Associates the rocket to which the Plate belongs to the
        _PlatePlots instance.

        Parameters
        ----------
        rocket : Rocket
            Rocket instance to which the plate belongs.
        """
        self._csys = rocket._csys
        self.center_of_dry_mass_position = rocket.center_of_dry_mass_position
        self.plots.rocket = rocket

    def info(self) -> None:
        """Prints a summary of the information stored in the plate object."""
        self.prints.all()

    def all_info(self) -> None:
        """Prints out all data and graphs available about the Plate."""
        self.plots.all()
        self.prints.all()

    @classmethod
    def from_dict(cls, data: dict) -> "Plate":
        """Creates a Plate instance from a dictionary containing the initialization
        parameters.

        Parameters
        ----------
        data : dict
            Dictionary containing plate constructor attributes.

        Returns
        -------
        plate : Plate
            Plate instance initialized with the specified parameters.
        """
        return cls(
            shape=data["shape"],
            dimensions=data["dimensions"],
            material=data["material"],
            thickness=data["thickness"],
            absolute_magnetic_permeability=data.get(
                "absolute_magnetic_permeability", None
            ),
            relative_magnetic_permeability=data.get(
                "relative_magnetic_permeability", None
            ),
            grid_spacing=data.get("grid_spacing", 0.001),
            z_points=data.get("z_points", 40),
            angular_points=data.get("angular_points", 70),
            name=data.get("name", "Plate"),
        )
