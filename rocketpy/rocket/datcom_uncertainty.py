"""DATCOM drag/CP engineering sensitivity with a Wu et al. drag-error floor.

Hanzhou Wu, Min Gao, Weidong Song, Zhao Jie and Yi Wang (2019), "Accuracy
Analysis of Aerodynamic Calculation of Two-dimensional Ballistic correction
Projectile based on Missile Datcom", IOP Conf. Ser.: Mater. Sci. Eng. 612,
032096. DOI: 10.1088/1757-899X/612/3/032096. Tables adapted under CC BY 3.0
(https://creativecommons.org/licenses/by/3.0/). The source validates one
canard projectile. Transferring its errors to other vehicles, symmetrizing
them, and holding boundary errors outside its envelope are assumptions,
not a statistical uncertainty model or a validation of RocketPy rockets.

The 20% drag floor and 2% nose-to-CP variation are engineering assumptions
motivated by J. T. White (1995), AIAA 95-1893, DOI 10.2514/6.1995-1893.
White studies supersonic CN/CP accuracy, not general CD uncertainty.
"""

from copy import copy

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rocketpy.rocket.datcom import DatcomSurface, zero_alpha_index


class Wu2019Errors:
    """Published signed errors, relative to wind-tunnel measurements.

    Mach and alpha (degrees) are linearly interpolated and clamped to the
    published envelope. Table 4 supplies Mach-only pitching-slope errors.
    CN's zero-alpha magnitude is the mean of the six nonzero-alpha magnitudes.
    No uncertainty is assigned to damping or lateral coefficients.
    """

    mach = np.array([0.8, 1.05, 1.2, 1.52, 2.01, 2.51, 3.02])
    alpha = np.array([-6, -4, -2, 0, 2, 4, 6])
    ca_error_percent = np.array(
        [
            [17.727, 16.380, 16.971, 14.127, 17.537, 21.429, 27.765],
            [-33.449, -34.727, -36.417, -35.593, -29.490, -23.369, -17.202],
            [-44.090, -47.563, -48.274, -47.981, -46.602, -43.404, -38.227],
            [-37.588, -39.327, -39.747, -40.246, -39.582, -37.759, -34.335],
            [-36.308, -38.934, -39.883, -40.076, -39.704, -37.219, -32.938],
            [-27.944, -31.351, -32.320, -32.401, -33.054, -31.042, -26.883],
            [-19.125, -23.563, -26.203, -28.598, -26.508, -23.681, -17.933],
        ]
    )
    cn_error_percent = np.array(
        [
            [20.943, 9.173, -30.941, 645.672, 95.265, 62.621, 54.258],
            [35.300, 20.209, -28.900, 554.288, 124.391, 91.398, 78.942],
            [30.573, 14.207, -25.887, 590.827, 89.558, 65.719, 59.669],
            [32.915, 27.855, 9.193, 69.492, 39.432, 36.843, 37.602],
            [13.922, 9.772, -4.739, 121.543, 26.513, 24.852, 26.066],
            [10.010, 10.531, 12.080, -23.744, -0.833, 3.548, 6.527],
            [4.373, 6.713, 9.702, -44.785, -13.970, -6.577, -2.769],
        ]
    )
    moment_mach = np.array([0.8, 0.95, 1.05, 1.15, 1.2, 1.52, 2.01, 2.51, 3.02])
    cma_error_percent = np.array(
        [-6.56, 81.47, 15.15, -26.15, -41.49, 19.47, -0.36, -15.23, -26.82]
    )

    def __init__(self):
        # e = (prediction - measurement) / measurement. A symmetric excursion
        # about the prediction must instead use |e / (1 + e)|. Applying the
        # printed 646% directly to DATCOM would use the wrong denominator.
        def relative(percent):
            return np.abs(percent / (100 + percent))

        normal = relative(self.cn_error_percent)
        # The source has canard-induced CN at alpha=0 and near-zero tunnel
        # measurements. Use the mean absolute discrepancy at nonzero AoAs
        # instead of transferring that ill-conditioned zero-incidence error.
        normal[:, self.alpha == 0] = normal[:, self.alpha != 0].mean(axis=1)[:, None]
        self._forces = RegularGridInterpolator(
            (self.mach, self.alpha),
            np.stack(
                [relative(self.ca_error_percent), normal],
                axis=-1,
            ),
        )
        self._moment = relative(self.cma_error_percent)

    def fractions(self, mach, alpha):
        """Return nonnegative native-relative CA, CN, CM-alpha excursions.

        ``alpha`` is in degrees. Boundary values are held outside the paper's
        envelope; they are not measured error bounds at those conditions.
        """
        m = float(np.clip(mach, self.mach[0], self.mach[-1]))
        a = float(np.clip(alpha, self.alpha[0], self.alpha[-1]))
        ca, cn = self._forces([m, a])[0]
        cma = np.interp(m, self.moment_mach, self._moment)
        return ca, cn, cma


class DatcomUncertaintySurface(DatcomSurface):
    """Flight-local CD/CP scenarios with unmodified native damping derivatives.

    Static wind-axis drag is varied by at least 20%, preserving perpendicular
    force. Inside Wu's Mach/alpha envelope, larger native-relative CA errors
    are used as a CD-error proxy. CP's distance from the nose varies by 2%.
    Upper means more drag and CP farther aft; lower reverses both. These two
    correlated scenarios are not guaranteed extrema of the trajectory.
    """

    @classmethod
    def from_model(cls, model, sign, *, nose_position):
        """Share native tables; provide the nose tip in the rocket's axial frame."""
        if sign not in (-1, 1) or not np.isfinite(nose_position):
            raise ValueError(
                "DATCOM sensitivity needs sign ±1 and a finite nose position"
            )
        result = cls.__new__(cls)
        result.__dict__ = copy(model.__dict__)
        result.sign = sign
        result.nose_position = float(nose_position)
        result.errors = Wu2019Errors()
        result.outside_alpha_envelope = False
        result.outside_mach_envelope = False
        result.outside_beta_envelope = False
        return result

    def drag_fraction(self, mach, alpha):
        """Native-relative CD excursion; alpha in degrees, 20% outside Wu data."""
        if (
            self.errors.mach[0] <= mach <= self.errors.mach[-1]
            and self.errors.alpha[0] <= alpha <= self.errors.alpha[-1]
        ):
            return max(0.2, float(self.errors.fractions(mach, alpha)[0]))
        return 0.2

    def _native_cp(self, powered, mach):
        """Native small-angle CP at the selected case's reference, in meters."""
        table = self.tables[powered]
        zero = zero_alpha_index(table[5])
        cna = np.interp(mach, table[4], table.derivatives["CNA"][:, zero])
        cma = np.interp(mach, table[4], table.derivatives["CMA"][:, zero])
        if abs(cna) < 1e-12:
            return float("nan")
        return self.references[powered] + self.csys * self.reference_length * cma / cna

    def coefficients(self, powered, mach, alpha, beta=0):
        """Vary static drag along the airflow and transfer its torque consistently.

        Alpha/beta are radians. The native static moment's residual couple is
        retained; added drag acts at the native small-angle CP. The CP shift
        transfers the new static force. Roll and all DAMP terms are unchanged.
        """
        values = super().coefficients(powered, mach, alpha, beta).copy()
        # Include the native linear sideslip contribution when there is no beta
        # grid. Leave that contribution for DatcomSurface to add exactly once.
        cy = values[3]
        if len(self._interpolators[powered].grid) == 2:
            cy += values[6] * np.degrees(beta)
        forces = np.array([cy, -values[0], -values[2]])
        direction = np.array([np.tan(beta), np.tan(alpha), 1.0])
        direction /= np.linalg.norm(direction)
        drag = -float(forces @ direction)
        delta = (
            -self.sign * self.drag_fraction(mach, np.degrees(alpha)) * drag * direction
        )
        changed = forces + delta
        values[0] -= delta[1]
        values[2] -= delta[2]
        values[3] += delta[0]
        values[-1] = -float(changed @ direction)

        native_cp = self._native_cp(powered, mach)
        if np.isfinite(native_cp):
            arm = (
                (native_cp - self.references[powered])
                * self.csys
                / self.reference_length
            )
            distance_from_nose = (self.nose_position - native_cp) * self.csys
            shift = -self.sign * 0.02 * distance_from_nose / self.reference_length
            # z cross (Fx, Fy, Fz) = (-Fy, Fx, 0). Static moments are
            # nondimensionalized by diameter, with no change of reference CG.
            values[1] -= arm * delta[1] + shift * changed[1]
            values[4] += arm * delta[0] + shift * changed[0]
        return values

    def small_angle_cp(self, mach):
        """Coast CP with ±2% nose-to-CP distance, in the user's axial frame."""
        native_cp = self._native_cp(False, mach)
        return native_cp - self.sign * 0.02 * (self.nose_position - native_cp)
