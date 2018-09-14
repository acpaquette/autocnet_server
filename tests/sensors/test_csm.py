import pytest
import math
from unittest.mock import MagicMock, Mock, patch

import pyproj
import numpy as np
from csmapi import csmapi

from autocnet_server.sensors import csm

@pytest.fixture
def mock_camera():
    mock_model = Mock(spec=csmapi.RasterGM)

    mock_iv = Mock(spec=csmapi.ImageVector)
    mock_iv.samp, mock_iv.line = 10, 10

    def create_ecef_coord(imageVector, z):
        ecef_mock = Mock(spec=csmapi.EcefCoord)

        ecef_mock.x = imageVector.samp
        ecef_mock.y = imageVector.line
        ecef_mock.z = z

        return ecef_mock

    mock_model.getImageSize = lambda : mock_iv
    mock_model.imageToGround = lambda imageVector, z: create_ecef_coord(imageVector, z)
    return mock_model

@pytest.fixture
def build_coord_map(x1_start, x1_stop, x2_start, x2_stop, size):
    x = np.linspace(x2_start, x2_stop, math.floor(size / 2))
    x = np.concatenate([np.linspace(x1_start, x1_stop, math.ceil(size / 2)), x])
    y = np.linspace(1, 0, size)

    lat_lon_boundary = [(i,y[0]) for i in x] + [(x[-1], i) for i in y[1:]] + [(i, y[-1]) for i in x[::-1][1:]] + [(x[0],i) for i in y[::-1][1:]]

    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)

    x_y_boundary = [(i,y[0]) for i in x] + [(x[-1], i) for i in y[1:]] + [(i, y[-1]) for i in x[::-1][1:]] + [(x[0],i) for i in y[::-1][1:]]

    return lat_lon_boundary, x_y_boundary

@pytest.mark.parametrize("x1_start, x1_stop, x2_start, x2_stop, size, n_points, expected", [
                        (179, 180, -180, -179, 11, 11, 2),
                        (170, 172, 173, 174, 11, 11, 1)
])
def test_footprint_gen(x1_start, x1_stop, x2_start, x2_stop, size, n_points, expected):
    camera = mock_camera()
    lat_lon_boundary, x_y_boundary = build_coord_map(x1_start, x1_stop, x2_start, x2_stop, size)

    def map_coords(arg1, arg2, x, y ,z):
        print(len(x), len(y))
        ll_coords = np.array([lat_lon_boundary[x_y_boundary.index((x[i], y[i]))] for i, val in enumerate(x)])
        lons, lats = ll_coords[:, 0], ll_coords[:, 1]
        return lons, lats, 0

    with patch.object(pyproj, 'transform', map_coords) as proj:
        footprint = csm.generate_latlon_footprint(camera, n_points=11, semi_major=10, semi_minor=10)

    assert expected == footprint.GetGeometryCount()
