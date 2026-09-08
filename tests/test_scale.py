from cozmo.reconstruct.scale import estimate_scale
from cozmo.reconstruct.tiles import spacings_to_metres_per_unit


def test_two_grout_periods_use_080_and_120():
    mpu, method = spacings_to_metres_per_unit([0.40, 0.60])
    assert method == "tile_0.80x1.20m"
    # 0.80/0.40 = 2, 1.20/0.60 = 2
    assert abs(mpu - 2.0) < 1e-9


def test_single_period_uses_80cm():
    mpu, method = spacings_to_metres_per_unit([0.5])
    assert method == "tile_0.80m"
    assert abs(mpu - 0.80 / 0.5) < 1e-9


def test_empty_spacings_are_none():
    assert spacings_to_metres_per_unit([]) is None


def test_door_3d_beats_tiles():
    scale = estimate_scale([], real_door_width_u=0.4, tile_spacings_u=[0.5])
    assert scale.method == "door_width_3d"
    assert abs(scale.metres_per_unit - 0.80 / 0.4) < 1e-9


def test_tiles_when_no_real_door():
    scale = estimate_scale([], real_door_width_u=None, tile_spacings_u=[0.40, 0.60])
    assert scale.method == "tile_0.80x1.20m"
    assert abs(scale.metres_per_unit - 2.0) < 1e-9


def test_hypothesized_door_does_not_set_metres():
    # Passing no real 3D door and no tiles must NOT invent 0.80 / 0.12 = 6.67 m walls.
    scale = estimate_scale([], real_door_width_u=None, tile_spacings_u=None)
    assert scale.method == "unscaled_scene_units"
    assert scale.metres_per_unit == 1.0
    assert scale.rel_error >= 0.5
