import pytest

from rocketpy.rocket.actuator.roll import RollActuator
from rocketpy.rocket.actuator.throttle import ThrottleActuator
from rocketpy.rocket.actuator.thrust_vector import (
    ThrustVectorActuator,
    ThrustVectorActuator2D,
)


class TestRollActuator:
    """Test suite for RollActuator class."""

    def test_initialization_defaults(self):
        """Test RollActuator initialization with default parameters."""
        actuator = RollActuator()
        assert actuator.name == "Roll Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (0, 0)
        assert actuator.roll_torque == 0.0
        assert actuator.actuator_rate_limit is None
        assert actuator.clamp is True

    def test_initialization_custom(self):
        """Test RollActuator initialization with custom parameters."""
        actuator = RollActuator(
            name="Custom Roll",
            demand_rate=50,
            max_roll_torque=10.0,
            torque_rate_limit=5.0,
            clamp=False,
            initial_roll_torque=2.0,
        )
        assert actuator.name == "Custom Roll"
        assert actuator.demand_rate == 50
        assert actuator.actuator_range == (-10.0, 10.0)
        assert actuator.roll_torque == 2.0
        assert actuator.actuator_rate_limit == 5.0
        assert actuator.clamp is False

    def test_roll_torque_property(self):
        """Test roll torque getter and setter."""
        actuator = RollActuator(max_roll_torque=10.0)
        actuator.roll_torque = 5.0
        assert actuator.roll_torque == 5.0

    def test_roll_torque_clamping(self):
        """Test roll torque clamping to range."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=True)
        actuator.actuator_output = 15.0
        assert actuator.roll_torque == 10.0
        actuator.actuator_output = -15.0
        assert actuator.roll_torque == -10.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = RollActuator(
            name="Test Roll",
            max_roll_torque=5.0,
            torque_rate_limit=2.0,
            initial_roll_torque=1.0,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test Roll"
        assert data["max_roll_torque"] == 5.0
        assert data["torque_rate_limit"] == 2.0
        assert data["initial_roll_torque"] == 1.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test Roll",
            "demand_rate": 50,
            "max_roll_torque": 5.0,
            "torque_rate_limit": 2.0,
            "clamp": True,
            "initial_roll_torque": 1.0,
            "roll_torque_time_constant": None,
        }
        actuator = RollActuator.from_dict(data)
        assert actuator.name == "Test Roll"
        assert actuator.demand_rate == 50
        assert actuator.roll_torque == 1.0

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = RollActuator(max_roll_torque=10.0, initial_roll_torque=2.0)
        actuator.roll_torque = 5.0
        actuator._reset()
        assert actuator.roll_torque == 2.0

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = RollActuator()
        # Just verify methods exist and don't raise exceptions
        actuator.info()
        actuator.all_info()


class TestThrottleActuator:
    """Test suite for ThrottleActuator class."""

    def test_initialization_defaults(self):
        """Test ThrottleActuator initialization with default parameters."""
        actuator = ThrottleActuator()
        assert actuator.name == "Throttle Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (0, 1)
        assert actuator.throttle == 0.0

    def test_initialization_custom(self):
        """Test ThrottleActuator initialization with custom parameters."""
        actuator = ThrottleActuator(
            name="Custom Throttle",
            demand_rate=50,
            max_throttle=0.8,
            throttle_rate_limit=0.1,
            initial_throttle=0.5,
        )
        assert actuator.name == "Custom Throttle"
        assert actuator.actuator_range == (0, 0.8)
        assert actuator.throttle == 0.5

    def test_throttle_property(self):
        """Test throttle getter and setter."""
        actuator = ThrottleActuator(max_throttle=1.0)
        actuator.throttle = 0.5
        assert actuator.throttle == 0.5

    def test_throttle_clamping(self):
        """Test throttle clamping to range."""
        actuator = ThrottleActuator(max_throttle=1.0, clamp=True)
        actuator.actuator_output = 1.5
        assert actuator.throttle == 1.0
        actuator.actuator_output = -0.5
        assert actuator.throttle == 0.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = ThrottleActuator(
            name="Test Throttle",
            max_throttle=0.8,
            throttle_rate_limit=0.1,
            initial_throttle=0.3,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test Throttle"
        assert data["max_throttle"] == 0.8
        assert data["throttle_rate_limit"] == 0.1
        assert data["initial_throttle"] == 0.3

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test Throttle",
            "demand_rate": 50,
            "max_throttle": 0.8,
            "throttle_rate_limit": 0.1,
            "clamp": True,
            "initial_throttle": 0.3,
            "throttle_time_constant": None,
        }
        actuator = ThrottleActuator.from_dict(data)
        assert actuator.name == "Test Throttle"
        assert actuator.throttle == 0.3

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = ThrottleActuator(initial_throttle=0.3)
        actuator.throttle = 0.7
        actuator._reset()
        assert actuator.throttle == 0.3

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = ThrottleActuator()
        actuator.info()
        actuator.all_info()


class TestThrustVectorActuator:
    """Test suite for ThrustVectorActuator class."""

    def test_initialization_defaults(self):
        """Test ThrustVectorActuator initialization with default parameters."""
        actuator = ThrustVectorActuator()
        assert actuator.name == "Thrust Vector Control"
        assert actuator.demand_rate == 100
        assert actuator.actuator_range == (-10, 10)
        assert actuator.gimbal_angle == 0.0

    def test_initialization_custom(self):
        """Test ThrustVectorActuator initialization with custom parameters."""
        actuator = ThrustVectorActuator(
            name="Custom TVC",
            demand_rate=50,
            max_gimbal_angle=15.0,
            gimbal_rate_limit=5.0,
            initial_gimbal_angle=5.0,
        )
        assert actuator.name == "Custom TVC"
        assert actuator.actuator_range == (-15.0, 15.0)
        assert actuator.gimbal_angle == 5.0

    def test_gimbal_angle_property(self):
        """Test gimbal angle getter and setter."""
        actuator = ThrustVectorActuator(max_gimbal_angle=10.0)
        actuator.gimbal_angle = 5.0
        assert actuator.gimbal_angle == 5.0

    def test_gimbal_angle_clamping(self):
        """Test gimbal angle clamping to range."""
        actuator = ThrustVectorActuator(max_gimbal_angle=10.0, clamp=True)
        actuator.actuator_output = 15.0
        assert actuator.gimbal_angle == 10.0
        actuator.actuator_output = -15.0
        assert actuator.gimbal_angle == -10.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        actuator = ThrustVectorActuator(
            name="Test TVC",
            max_gimbal_angle=8.0,
            gimbal_rate_limit=3.0,
            initial_gimbal_angle=2.0,
        )
        data = actuator.to_dict()
        assert data["name"] == "Test TVC"
        assert data["max_gimbal_angle"] == 8.0
        assert data["gimbal_rate_limit"] == 3.0
        assert data["initial_gimbal_angle"] == 2.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "Test TVC",
            "demand_rate": 50,
            "max_gimbal_angle": 8.0,
            "gimbal_rate_limit": 3.0,
            "clamp": True,
            "initial_gimbal_angle": 2.0,
            "gimbal_time_constant": None,
        }
        actuator = ThrustVectorActuator.from_dict(data)
        assert actuator.name == "Test TVC"
        assert actuator.gimbal_angle == 2.0

    def test_reset(self):
        """Test actuator reset functionality."""
        actuator = ThrustVectorActuator(initial_gimbal_angle=2.0)
        actuator.gimbal_angle = 5.0
        actuator._reset()
        assert actuator.gimbal_angle == 2.0

    def test_info_methods(self):
        """Test info and all_info methods."""
        actuator = ThrustVectorActuator()
        actuator.info()
        actuator.all_info()


class TestThrustVectorActuator2D:
    """Test suite for ThrustVectorActuator2D class."""

    def test_initialization_defaults(self):
        """Test ThrustVectorActuator2D initialization with default parameters."""
        actuator = ThrustVectorActuator2D()
        assert actuator.gimbal_angle_x == 0.0
        assert actuator.gimbal_angle_y == 0.0
        assert actuator.gimbal_angles == (0.0, 0.0)

    def test_initialization_custom(self):
        """Test ThrustVectorActuator2D initialization with custom parameters."""
        actuator = ThrustVectorActuator2D(
            name="Custom 2D TVC",
            max_gimbal_angle=15.0,
            initial_gimbal_angle=5.0,
        )
        assert actuator.x.name == "Custom 2D TVC X-axis"
        assert actuator.y.name == "Custom 2D TVC Y-axis"
        assert actuator.gimbal_angles == (5.0, 5.0)

    def test_gimbal_angle_x_property(self):
        """Test gimbal angle X getter and setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_x = 5.0
        assert actuator.gimbal_angle_x == 5.0

    def test_gimbal_angle_y_property(self):
        """Test gimbal angle Y getter and setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_y = 7.0
        assert actuator.gimbal_angle_y == 7.0

    def test_gimbal_angles_property_get(self):
        """Test gimbal angles tuple getter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angle_x = 3.0
        actuator.gimbal_angle_y = 4.0
        assert actuator.gimbal_angles == (3.0, 4.0)

    def test_gimbal_angles_property_set(self):
        """Test gimbal angles tuple setter."""
        actuator = ThrustVectorActuator2D()
        actuator.gimbal_angles = (5.0, 7.0)
        assert actuator.gimbal_angle_x == 5.0
        assert actuator.gimbal_angle_y == 7.0
        assert actuator.gimbal_angles == (5.0, 7.0)

    def test_independent_axes(self):
        """Test that X and Y axes are independent."""
        actuator = ThrustVectorActuator2D(max_gimbal_angle=10.0)
        actuator.gimbal_angle_x = 5.0
        actuator.gimbal_angle_y = -3.0
        assert actuator.gimbal_angle_x == 5.0
        assert actuator.gimbal_angle_y == -3.0
        assert actuator.gimbal_angles == (5.0, -3.0)

    def test_clamping_individual_axes(self):
        """Test clamping on individual axes."""
        actuator = ThrustVectorActuator2D(max_gimbal_angle=10.0, clamp=True)
        # Test X axis clamping
        actuator.x.actuator_output = 15.0
        assert actuator.gimbal_angle_x == 10.0
        # Test Y axis clamping
        actuator.y.actuator_output = -15.0
        assert actuator.gimbal_angle_y == -10.0


class TestActuatorDynamics:
    """Test suite for actuator dynamics (rate limiting, time constants)."""

    def test_rate_limiting_roll(self):
        """Test rate limiting on roll actuator."""
        actuator = RollActuator(
            max_roll_torque=10.0,
            demand_rate=100,
            torque_rate_limit=5.0,
        )
        # Change should be limited
        actuator.actuator_output = 10.0  # Try to jump to 10
        # At 100 Hz demand rate: max_change = 5.0 / 100 = 0.05
        # So output should be clamped to 0.05
        assert actuator.roll_torque <= 0.1  # Small due to rate limit

    def test_no_rate_limiting_when_none(self):
        """Test that no rate limiting occurs when set to None."""
        actuator = RollActuator(
            max_roll_torque=10.0,
            torque_rate_limit=None,
        )
        actuator.actuator_output = 5.0
        assert actuator.roll_torque == 5.0

    def test_time_constant_iir_filter(self):
        """Test IIR filter behavior with time constant."""
        actuator = ThrottleActuator(
            max_throttle=1.0,
            demand_rate=100,
            throttle_time_constant=0.1,
        )
        # With time constant, output should be filtered
        # alpha = Ts / (tau + Ts) = 0.01 / (0.1 + 0.01) ≈ 0.0909
        initial_output = actuator.throttle
        actuator.actuator_output = 1.0
        filtered_output = actuator.throttle
        # Output should be between initial and 1.0 due to filtering
        assert initial_output < filtered_output < 1.0


class TestActuatorValidation:
    """Test suite for actuator parameter validation."""

    def test_invalid_demand_rate_negative(self):
        """Test that negative demand rate raises assertion error."""
        with pytest.raises(AssertionError):
            RollActuator(demand_rate=-1)

    def test_invalid_range(self):
        """Test that invalid range raises assertion error."""
        with pytest.raises(AssertionError):
            RollActuator(
                max_roll_torque=-5
            )  # This creates range (5, -5) which is invalid

    def test_invalid_time_constant_negative(self):
        """Test that negative time constant raises assertion error."""
        with pytest.raises(AssertionError):
            ThrottleActuator(throttle_time_constant=-0.1)

    def test_invalid_rate_limit_negative(self):
        """Test that negative rate limit raises assertion error."""
        with pytest.raises(AssertionError):
            ThrustVectorActuator(gimbal_rate_limit=-1.0)


class TestActuatorWarnings:
    """Test suite for actuator warning conditions."""

    def test_clamp_false_no_clamping(self):
        """Test that output is not clamped when clamp=False."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=False)
        actuator.actuator_output = 15.0
        # Output should not be clamped when clamp=False
        assert actuator.roll_torque == 15.0

    def test_clamping_applied(self):
        """Test that clamping is applied when clamp=True."""
        actuator = RollActuator(max_roll_torque=10.0, clamp=True)
        actuator.actuator_output = 15.0
        # Output should be clamped to range
        assert actuator.roll_torque == 10.0
