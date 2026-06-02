from copy import deepcopy


class Commands:
    """Command API exposed to hook trigger/callback callables."""

    def __init__(self):
        self._default_results = {
            "exact_time": None,
            "exact_state": None,
            "disable": None,
            "rollback": False,
            "new_events": [],
            "disable_events": [],
            "new_controllers": [],
            "disable_controllers": [],
            "new_derivative": None,
            "new_derivative_set": False,
            "new_flight_phase": None,
            "new_flight_phase_name": None,
            "new_flight_phase_lag": 0,
            "new_flight_phase_parachute": None,
            "terminate_flight": False,
            "terminate_phase_name": None,
        }
        self.results = deepcopy(self._default_results)

    def reset(self):
        self.results = deepcopy(self._default_results)

    def disable(self):
        self.results["disable"] = True

    def enable(self):
        self.results["disable"] = False

    def add_event(self, event):
        self.results["new_events"].append(event)

    def disable_event(self, event):
        self.results["disable_events"].append(event)

    def add_controller(self, controller):
        self.results["new_controllers"].append(controller)

    def disable_controller(self, controller):
        self.results["disable_controllers"].append(controller)

    def set_derivative(self, derivative):
        self.results["new_derivative"] = derivative
        self.results["new_derivative_set"] = True

    def start_flight_phase(self, phase_name=None, lag=0, parachute=None):
        self.results["new_flight_phase"] = True
        self.results["new_flight_phase_name"] = phase_name
        self.results["new_flight_phase_lag"] = lag
        self.results["new_flight_phase_parachute"] = parachute

    def terminate_flight(self):
        self.results["terminate_flight"] = True
