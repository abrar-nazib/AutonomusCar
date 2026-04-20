from autocar.control.mixer import DifferentialMixer


def test_straight_ahead_both_sides_equal():
    mix = DifferentialMixer(base_speed=100, max_speed=200)
    cmd = mix.mix(throttle=1.0, steering=0.0)
    assert cmd.left == cmd.right == 100


def test_steer_right_reduces_right_side():
    mix = DifferentialMixer(base_speed=100, max_speed=200)
    cmd = mix.mix(throttle=1.0, steering=1.0)
    assert cmd.right > cmd.left


def test_output_clamped_to_max_speed():
    mix = DifferentialMixer(base_speed=200, max_speed=200)
    cmd = mix.mix(throttle=1.0, steering=1.0)
    assert -200 <= cmd.left <= 200
    assert -200 <= cmd.right <= 200
