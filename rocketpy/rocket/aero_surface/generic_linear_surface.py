from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.mathutils import Function


class GenericLinearSurface(GenericSurface):
    """Class that defines a generic linear aerodynamic surface. This class is
    used to define aerodynamic surfaces that have aerodynamic coefficients
    defined as linear functions of the coefficients derivatives."""

    def __init__(
        self,
        reference_area,
        reference_length,
        cL_0=0,
        cL_alpha=0,
        cL_beta=0,
        cL_p=0,
        cL_q=0,
        cL_r=0,
        cQ_0=0,
        cQ_alpha=0,
        cQ_beta=0,
        cQ_p=0,
        cQ_q=0,
        cQ_r=0,
        cD_alpha=0,
        cD_beta=0,
        cD_p=0,
        cD_q=0,
        cD_r=0,
        cm_0=0,
        cm_alpha=0,
        cm_beta=0,
        cm_p=0,
        cm_q=0,
        cm_r=0,
        cn_0=0,
        cn_alpha=0,
        cn_beta=0,
        cn_p=0,
        cn_q=0,
        cn_r=0,
        cl_0=0,
        cl_alpha=0,
        cl_beta=0,
        cl_p=0,
        cl_q=0,
        cl_r=0,
        center_of_pressure=(0, 0, 0),
        name="Generic Linear Surface",
    ):
        """Initializes the generic aerodynamic surface object.

        Important
        ---------
        The coefficients can be defined as a CSV file or as a callable function.
        The function must have 4 input arguments in the form (alpha, beta, mach,
        height). The CSV file must have a header with the following columns:
        'alpha', 'beta', 'mach', 'height', 'pitch_rate', 'yaw_rate', 'roll_rate'
        and the last column must be the coefficient value, which can have any
        name. Not all columns are required, but at least one independent
        variable and the coefficient value are required.

        See Also
        --------
        LINK TO DOCS

        Parameters
        ----------
        reference_area : int, float
            Reference area of the aerodynamic surface. Has the unit of meters
            squared. Commonly defined as the rocket's cross-sectional area.
        reference_length : int, float
            Reference length of the aerodynamic surface. Has the unit of meters.
            Commonly defined as the rocket's diameter.
        cL_0 : callable, str, optional
            Coefficient of lift at zero angle of attack. Default is 0.
        cL_alpha : callable, str, optional
            Coefficient of lift derivative with respect to angle of attack.
            Default is 0.
        cL_beta : callable, str, optional
            Coefficient of lift derivative with respect to sideslip angle.
            Default is 0.
        cL_p : callable, str, optional
            Coefficient of lift derivative with respect to roll rate.
            Default is 0.
        cL_q : callable, str, optional
            Coefficient of lift derivative with respect to pitch rate.
            Default is 0.
        cL_r : callable, str, optional
            Coefficient of lift derivative with respect to yaw rate.
            Default is 0.
        cQ_0 : callable, str, optional
            Coefficient of pitch moment at zero angle of attack.
            Default is 0.
        cQ_alpha : callable, str, optional
            Coefficient of pitch moment derivative with respect to angle of
            attack. Default is 0.
        cQ_beta : callable, str, optional
            Coefficient of pitch moment derivative with respect to sideslip
            angle. Default is 0.
        cQ_p : callable, str, optional
            Coefficient of pitch moment derivative with respect to roll rate.
            Default is 0.
        cQ_q : callable, str, optional
            Coefficient of pitch moment derivative with respect to pitch rate.
            Default is 0.
        cQ_r : callable, str, optional
            Coefficient of pitch moment derivative with respect to yaw rate.
            Default is 0.
        cD_alpha : callable, str, optional
            Coefficient of drag derivative with respect to angle of attack.
            Default is 0.
        cD_beta : callable, str, optional
            Coefficient of drag derivative with respect to sideslip angle.
            Default is 0.
        cD_p : callable, str, optional
            Coefficient of drag derivative with respect to roll rate.
            Default is 0.
        cD_q : callable, str, optional
            Coefficient of drag derivative with respect to pitch rate.
            Default is 0.
        cD_r : callable, str, optional
            Coefficient of drag derivative with respect to yaw rate.
            Default is 0.
        cm_0 : callable, str, optional
            Coefficient of pitch moment at zero angle of attack.
            Default is 0.
        cm_alpha : callable, str, optional
            Coefficient of pitch moment derivative with respect to angle of
            attack. Default is 0.
        cm_beta : callable, str, optional
            Coefficient of pitch moment derivative with respect to sideslip
            angle. Default is 0.
        cm_p : callable, str, optional
            Coefficient of pitch moment derivative with respect to roll rate.
            Default is 0.
        cm_q : callable, str, optional
            Coefficient of pitch moment derivative with respect to pitch rate.
            Default is 0.
        cm_r : callable, str, optional
            Coefficient of pitch moment derivative with respect to yaw rate.
            Default is 0.
        cn_0 : callable, str, optional
            Coefficient of yaw moment at zero angle of attack.
            Default is 0.
        cn_alpha : callable, str, optional
            Coefficient of yaw moment derivative with respect to angle of
            attack. Default is 0.
        cn_beta : callable, str, optional
            Coefficient of yaw moment derivative with respect to sideslip angle.
            Default is 0.
        cn_p : callable, str, optional
            Coefficient of yaw moment derivative with respect to roll rate.
            Default is 0.
        cn_q : callable, str, optional
            Coefficient of yaw moment derivative with respect to pitch rate.
            Default is 0.
        cn_r : callable, str, optional
            Coefficient of yaw moment derivative with respect to yaw rate.
            Default is 0.
        cl_0 : callable, str, optional
            Coefficient of roll moment at zero angle of attack.
            Default is 0.
        cl_alpha : callable, str, optional
            Coefficient of roll moment derivative with respect to angle of
            attack. Default is 0.
        cl_beta : callable, str, optional
            Coefficient of roll moment derivative with respect to sideslip
            angle. Default is 0.
        cl_p : callable, str, optional
            Coefficient of roll moment derivative with respect to roll rate.
            Default is 0.
        cl_q : callable, str, optional
            Coefficient of roll moment derivative with respect to pitch rate.
            Default is 0.
        cl_r : callable, str, optional
            Coefficient of roll moment derivative with respect to yaw rate.
            Default is 0.
        center_of_pressure : tuple, optional
            Application point of the aerodynamic forces and moments. The
            center of pressure is defined in the local coordinate system of the
            aerodynamic surface. The default value is (0, 0, 0).
        name : str
            Name of the aerodynamic surface. Default is 'GenericSurface'.
        """
        self.cL_0 = self._process_input(cL_0)
        self.cL_alpha = self._process_input(cL_alpha)
        self.cL_beta = self._process_input(cL_beta)
        self.cL_p = self._process_input(cL_p)
        self.cL_q = self._process_input(cL_q)
        self.cL_r = self._process_input(cL_r)
        self.cQ_0 = self._process_input(cQ_0)
        self.cQ_alpha = self._process_input(cQ_alpha)
        self.cQ_beta = self._process_input(cQ_beta)
        self.cQ_p = self._process_input(cQ_p)
        self.cQ_q = self._process_input(cQ_q)
        self.cQ_r = self._process_input(cQ_r)
        self.cD_alpha = self._process_input(cD_alpha)
        self.cD_beta = self._process_input(cD_beta)
        self.cD_p = self._process_input(cD_p)
        self.cD_q = self._process_input(cD_q)
        self.cD_r = self._process_input(cD_r)
        self.cl_0 = self._process_input(cl_0)
        self.cl_alpha = self._process_input(cl_alpha)
        self.cl_beta = self._process_input(cl_beta)
        self.cl_p = self._process_input(cl_p)
        self.cl_q = self._process_input(cl_q)
        self.cl_r = self._process_input(cl_r)
        self.cm_0 = self._process_input(cm_0)
        self.cm_alpha = self._process_input(cm_alpha)
        self.cm_beta = self._process_input(cm_beta)
        self.cm_p = self._process_input(cm_p)
        self.cm_q = self._process_input(cm_q)
        self.cm_r = self._process_input(cm_r)
        self.cn_0 = self._process_input(cn_0)
        self.cn_alpha = self._process_input(cn_alpha)
        self.cn_beta = self._process_input(cn_beta)
        self.cn_p = self._process_input(cn_p)
        self.cn_q = self._process_input(cn_q)
        self.cn_r = self._process_input(cn_r)

        self.compute_all_coefficients()

        super().__init__(
            reference_area=reference_area,
            reference_length=reference_length,
            cL=self.cL,
            cQ=self.cQ,
            cD=self.cD,
            cm=self.cm,
            cn=self.cn,
            cl=self.cl,
            center_of_pressure=center_of_pressure,
            name=name,
        )

    def compute_linear_coefficient(
        self, c_0, c_alpha, c_beta, c_p, c_q, c_r, coeff_name
    ):

        def total_coefficient(alpha, beta, mach, height, p, q, r):
            return (
                c_0(alpha, beta, mach, height)
                + c_alpha(alpha, beta, mach, height) * alpha
                + c_beta(alpha, beta, mach, height) * beta
                + self.reference_length
                / (2 * self.stream_speed)
                * c_p(alpha, beta, mach, height)
                * p
                + self.reference_length
                / (2 * self.stream_speed)
                * c_q(alpha, beta, mach, height)
                * q
                + self.reference_length
                / (2 * self.stream_speed)
                * c_r(alpha, beta, mach, height)
                * r
            )

        return Function(
            total_coefficient,
            ['alpha', 'beta', 'mach', 'height', 'p', 'q', 'r'],
            [coeff_name],
        )

    def compute_all_coefficients(self):
        self.cL = self.compute_linear_coefficient(
            self.cL_0,
            self.cL_alpha,
            self.cL_beta,
            self.cL_p,
            self.cL_q,
            self.cL_r,
            'cL',
        )
        self.cQ = self.compute_linear_coefficient(
            self.cQ_0,
            self.cQ_alpha,
            self.cQ_beta,
            self.cQ_p,
            self.cQ_q,
            self.cQ_r,
            'cQ',
        )
        self.cD = self.compute_linear_coefficient(
            self.cD_0,
            self.cD_alpha,
            self.cD_beta,
            self.cD_p,
            self.cD_q,
            self.cD_r,
            'cD',
        )
        self.cm = self.compute_linear_coefficient(
            self.cm_0,
            self.cm_alpha,
            self.cm_beta,
            self.cm_p,
            self.cm_q,
            self.cm_r,
            'cm',
        )
        self.cn = self.compute_linear_coefficient(
            self.cn_0,
            self.cn_alpha,
            self.cn_beta,
            self.cn_p,
            self.cn_q,
            self.cn_r,
            'cn',
        )
        self.cl = self.compute_linear_coefficient(
            self.cl_0,
            self.cl_alpha,
            self.cl_beta,
            self.cl_p,
            self.cl_q,
            self.cl_r,
            'cl',
        )
