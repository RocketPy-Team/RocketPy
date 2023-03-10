from .DispersionModel import DispersionModel
from .mc_aero_surfaces import McEllipticalFins, McNoseCone, McTail, McTrapezoidalFins
from .mc_environment import McEnvironment
from .mc_flight import McFlight
from .mc_parachute import McParachute
from .mc_rocket import McRocket
from .mc_solid_motor import McSolidMotor

# TODO: the set_attr() is defined 8 times across the monte carlo classes. It should be defined only once and imported from a single location. Try to use the tools.py file for this.
# TODO: finish documentation of all the monte carlo classes
