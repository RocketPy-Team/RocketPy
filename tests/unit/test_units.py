import numpy as np
import pytest

from rocketpy import Function
from rocketpy.units import conversion_factor, convert_temperature, convert_units


class TestConvertTemperature:
    """Tests for the convert_temperature function."""

    def test_convert_temperature_same_unit(self):
        assert convert_temperature(300, "K", "K") == 300
        assert convert_temperature(27, "degC", "degC") == 27
        assert convert_temperature(80, "degF", "degF") == 80

    def test_convert_temperature_kelvin_to_celsius(self):
        assert convert_temperature(300, "K", "degC") == pytest.approx(26.85, rel=1e-2)

    def test_convert_temperature_kelvin_to_fahrenheit(self):
        assert convert_temperature(300, "K", "degF") == pytest.approx(80.33, rel=1e-2)

    def test_convert_temperature_celsius_to_kelvin(self):
        assert convert_temperature(27, "degC", "K") == pytest.approx(300.15, rel=1e-2)

    def test_convert_temperature_celsius_to_fahrenheit(self):
        assert convert_temperature(27, "degC", "degF") == pytest.approx(80.6, rel=1e-2)

    def test_convert_temperature_fahrenheit_to_kelvin(self):
        assert convert_temperature(80, "degF", "K") == pytest.approx(299.817, rel=1e-2)

    def test_convert_temperature_fahrenheit_to_celsius(self):
        assert convert_temperature(80, "degF", "degC") == pytest.approx(26.67, rel=1e-2)

    def test_convert_temperature_invalid_conversion(self):
        with pytest.raises(ValueError):
            convert_temperature(300, "K", "invalid_unit")
        with pytest.raises(ValueError):
            convert_temperature(300, "invalid_unit", "K")


class TestConversionFactor:
    """Tests for the conversion_factor function."""

    def test_conversion_factor_same_unit(self):
        assert conversion_factor("m", "m") == 1
        assert conversion_factor("ft", "ft") == 1
        assert conversion_factor("s", "s") == 1

    def test_conversion_factor_m_to_ft(self):
        assert conversion_factor("m", "ft") == pytest.approx(3.28084, rel=1e-2)

    def test_conversion_factor_ft_to_m(self):
        assert conversion_factor("ft", "m") == pytest.approx(0.3048, rel=1e-2)

    def test_conversion_factor_s_to_min(self):
        assert conversion_factor("s", "min") == pytest.approx(1 / 60, rel=1e-2)

    def test_conversion_factor_min_to_s(self):
        assert conversion_factor("min", "s") == pytest.approx(60, rel=1e-2)

    def test_conversion_factor_invalid_conversion(self):
        with pytest.raises(ValueError):
            conversion_factor("m", "invalid_unit")
        with pytest.raises(ValueError):
            conversion_factor("invalid_unit", "m")


class TestConvertUnits:
    """Tests for the convert_units function."""

    @pytest.mark.parametrize(
        "unit, base_unit, value_in_base",
        [
            ("mm", "m", 1e-3),
            ("cm", "m", 1e-2),
            ("dm", "m", 1e-1),
            ("dam", "m", 1e1),
            ("hm", "m", 1e2),
            ("km", "m", 1e3),
            ("ft", "m", 0.3048),
            ("in", "m", 0.0254),
            ("mi", "m", 1609.344),
            ("nmi", "m", 1852),
            ("yd", "m", 0.9144),
            ("km/h", "m/s", 1 / 3.6),
            ("knot", "m/s", 1852 / 3600),
            ("mph", "m/s", 1609.344 / 3600),
            ("ft/s", "m/s", 0.3048),
            ("gs", "m/s^2", 9.80665),
            ("ft/s^2", "m/s^2", 0.3048),
            ("hPa", "Pa", 1e2),
            ("kPa", "Pa", 1e3),
            ("MPa", "Pa", 1e6),
            ("bar", "Pa", 1e5),
            ("atm", "Pa", 101325),
            ("mmHg", "Pa", 133.322387415),
            ("inHg", "Pa", 3386.389),
            ("min", "s", 60),
            ("h", "s", 3600),
            ("d", "s", 86400),
            ("mg", "kg", 1e-6),
            ("g", "kg", 1e-3),
            ("lb", "kg", 0.45359237),
            ("deg", "rad", np.pi / 180),
            ("grad", "rad", np.pi / 200),
        ],
    )
    def test_convert_units_matches_unit_definitions(
        self, unit, base_unit, value_in_base
    ):
        """One of each unit should convert to its defined value in the base
        unit, and back. The references are the exact SI definitions, so the
        tolerance only allows for the rounded pound and mercury constants."""
        assert convert_units(1, unit, base_unit) == pytest.approx(
            value_in_base, rel=1e-5
        )
        assert convert_units(value_in_base, base_unit, unit) == pytest.approx(
            1, rel=1e-5
        )

    def test_convert_units_same_unit(self):
        assert convert_units(300, "K", "K") == 300
        assert convert_units(27, "degC", "degC") == 27
        assert convert_units(80, "degF", "degF") == 80

    def test_convert_units_kelvin_to_celsius(self):
        assert convert_units(300, "K", "degC") == pytest.approx(26.85, rel=1e-2)

    def test_convert_units_kelvin_to_fahrenheit(self):
        assert convert_units(300, "K", "degF") == pytest.approx(80.33, rel=1e-2)

    def test_convert_units_kilogram_to_pound(self):
        assert convert_units(1, "kg", "lb") == pytest.approx(2.20462, rel=1e-2)

    def test_convert_units_kilometer_to_mile(self):
        assert convert_units(1, "km", "mi") == pytest.approx(0.621371, rel=1e-2)

    def test_convert_units_function_input_axis(self):
        function = Function(
            np.array([[0.0, 0.0], [60.0, 100.0]]),
            inputs="Time (s)",
            outputs="Distance (m)",
            interpolation="linear",
            extrapolation="zero",
        )

        converted = convert_units(function, "s", "min", axis=0)

        np.testing.assert_allclose(
            converted.get_source(), np.array([[0.0, 0.0], [1.0, 100.0]])
        )
        assert converted.__inputs__ == ["Time (min)"]
        assert converted.__outputs__ == ["Distance (m)"]
        assert converted.__interpolation__ == "linear"
        assert converted.__extrapolation__ == "zero"

    def test_convert_units_function_temperature_output(self):
        function = Function(
            np.array([[0.0, 273.15], [1.0, 373.15]]),
            inputs="Time (s)",
            outputs="Temperature (K)",
            interpolation="linear",
            extrapolation="constant",
        )

        converted = convert_units(function, "K", "degC")

        np.testing.assert_allclose(
            converted.get_source(), np.array([[0.0, 0.0], [1.0, 100.0]])
        )
        assert converted.__inputs__ == ["Time (s)"]
        assert converted.__outputs__ == ["Temperature (degC)"]
        assert converted.__interpolation__ == "linear"
        assert converted.__extrapolation__ == "constant"
