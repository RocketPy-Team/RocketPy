`for006.dat` is native Missile DATCOM revision 3/99 output for a simple,
unblunted von Karman nose, cylindrical body, four HEX fins and a conical
boattail. The echoed deck specifies the geometry, two Mach points and three
signed incidence angles. DAMP and full base drag are enabled; protuberances
and plume interaction are absent.

This fixture checks parsing and coefficient-channel completeness. It is not
an independently validated aerodynamic reference. No executable is included.


`manual_example_for005.dat` independently transcribes the worked two-fin-set
example in the Missile DATCOM Users Guide, revision 5/97, Figures 19–21
(printed pages 65–67). It uses inches, Mach 2.36 and the published geometry,
not RocketPy's deck generator. The optional native regression compares the
published CN/CM/CA values and verifies that PART retains total-vehicle damping.
The local 3/99 executable reproduces the published coefficient table.
