from autocar.control.pid import PID


def test_proportional_only_tracks_sign_of_error():
    pid = PID(kp=1.0, ki=0.0, kd=0.0)
    assert pid.update(0.5, dt=0.1) > 0
    pid.reset()
    assert pid.update(-0.5, dt=0.1) < 0


def test_output_is_clamped_to_limit():
    pid = PID(kp=10.0, ki=0.0, kd=0.0, output_limit=1.0)
    assert pid.update(5.0, dt=0.1) == 1.0
    pid.reset()
    assert pid.update(-5.0, dt=0.1) == -1.0


def test_zero_dt_returns_zero():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    assert pid.update(1.0, dt=0.0) == 0.0
