"""Geometry projections for DATCOM-specific panel placement."""

import numpy as np

from rocketpy.rocket.aero_surface.nose_cone import NoseCone


def draw_datcom_fins(surface, ax, position, drawn_surfaces, vis_args, plane, csys):
    """Draw a true-view panel outline at PHIF/GAM and fixed incidence.

    The incidence hinge is DATCOM's default half-root-chord line; SKEW is zero.
    Root attachment radius is set from the assembled body during preparation.
    """
    aft = np.asarray(surface.shape_vec[0])
    span = np.asarray(surface.shape_vec[1])
    radius = getattr(surface, "_datcom_attachment_radius", surface.rocket_radius)
    hinge = surface.root_chord / 2
    component = 0 if plane == "xz" else 1
    for phi, gam, delta in zip(
        surface.panel_angles, surface.dihedral, surface.datcom_incidence
    ):
        phi, gam, delta = np.radians([phi, gam, delta])
        root = radius * np.array([np.sin(phi), np.cos(phi)])
        direction = np.array([np.sin(phi + gam), np.cos(phi + gam)])
        tangent = np.array([np.cos(phi + gam), -np.sin(phi + gam)])
        transverse = root[:, None] + direction[:, None] * span
        transverse -= tangent[:, None] * ((aft - hinge) * np.sin(delta))
        axial = position - csys * (hinge + (aft - hinge) * np.cos(delta))
        x = np.append(axial, axial[0])
        y = np.append(transverse[component], transverse[component, 0])
        ax.plot(x, y, color=vis_args["fins"], linewidth=vis_args["line_width"])
    body = getattr(surface, "_datcom_drawing_body", None)
    if body is not None:
        # Fins are overlays, not diameter transitions. In particular a fin on
        # the boattail must not introduce a fictitious cylindrical tube there.
        if not body["has_tail"]:
            for index, (component, origin, body_radius, end) in enumerate(
                drawn_surfaces
            ):
                if isinstance(component, NoseCone):
                    drawn_surfaces[index] = (
                        component,
                        origin,
                        body_radius,
                        body["end"],
                    )
        return
    drawn_surfaces.append(
        (surface, position, radius, position - csys * surface.root_chord)
    )
