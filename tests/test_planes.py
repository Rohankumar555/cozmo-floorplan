import numpy as np

from cozmo.reconstruct.planes import polygon_area, sequential_planes, floor_polygon_from_planes


def test_box_planes_make_a_polygon():
    rng = np.random.default_rng(1)
    # Axis-aligned room: x,z in [0,2]x[0,3], y up [0,1]
    pts = []
    for z in (0.0, 1.0):
        xx, zz = np.meshgrid(np.linspace(0, 2, 30), np.linspace(0, 3, 30))
        pts.append(np.stack([xx.ravel(), np.full(xx.size, z), zz.ravel()], axis=1))
    xx, yy = np.meshgrid(np.linspace(0, 2, 30), np.linspace(0, 1, 12))
    pts.append(np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1))
    pts.append(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 3.0)], axis=1))
    zz, yy = np.meshgrid(np.linspace(0, 3, 30), np.linspace(0, 1, 12))
    pts.append(np.stack([np.zeros(zz.size), yy.ravel(), zz.ravel()], axis=1))
    pts.append(np.stack([np.full(zz.size, 2.0), yy.ravel(), zz.ravel()], axis=1))
    cloud = np.concatenate(pts) + rng.normal(0, 0.005, size=(sum(len(p) for p in pts), 3))
    planes = sequential_planes(cloud, thresh=0.03, rng=rng)
    assert len(planes) >= 3
    poly, ceiling, _ = floor_polygon_from_planes(planes, cloud)
    assert len(poly) >= 3
    assert ceiling > 0.5
    assert polygon_area(poly) > 1.0
