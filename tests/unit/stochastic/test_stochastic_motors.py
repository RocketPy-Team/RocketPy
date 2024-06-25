from rocketpy.motors import GenericMotor


def test_stochastic_generic_motor_create_object(stochastic_generic_motor):
    obj = stochastic_generic_motor.create_object()
    assert isinstance(obj, GenericMotor)
