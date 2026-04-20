from autocar.comms.uart import encode_command
from autocar.control.mixer import MotorCommand


def test_encode_is_ascii_line():
    assert encode_command(MotorCommand(left=120, right=-80)) == b"M 120 -80\n"


def test_encode_clamps_to_pwm_range():
    assert encode_command(MotorCommand(left=500, right=-500)) == b"M 255 -255\n"
