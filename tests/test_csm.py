import datetime
import json
from unittest import mock
import math
from unittest.mock import MagicMock, Mock, patch

import pytest
import pyproj
import numpy as np
from csmapi import csmapi

from autocnet_server.sensors import csm
import pvl

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
def build_coord_map(request):
    print(request.param)
    x1_start, x1_stop, x2_start, x2_stop, size = request.param
    x = np.linspace(x2_start, x2_stop, math.floor(size / 2))
    x = np.concatenate([np.linspace(x1_start, x1_stop, math.ceil(size / 2)), x])
    y = np.linspace(1, 0, size)

    lat_lon_boundary = [(i,y[0]) for i in x] + [(x[-1], round(i, 2)) for i in y[1:]] + [(i, y[-1]) for i in x[::-1][1:]] + [(x[0], round(i, 2)) for i in y[::-1][1:]]

    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)

    x_y_boundary = [(i,y[0]) for i in x] + [(x[-1], i) for i in y[1:]] + [(i, y[-1]) for i in x[::-1][1:]] + [(x[0],i) for i in y[::-1][1:]]

    return lat_lon_boundary, x_y_boundary

@pytest.mark.parametrize('build_coord_map', ([179, 180, -180, -179, 11],), indirect = True)
@pytest.mark.parametrize('n_points, expected', [(11, 2)])
def test_latlon_prime_footprint(build_coord_map, n_points, expected, mock_camera):
    lat_lon_boundary, x_y_boundary = build_coord_map

    def map_coords(arg1, arg2, x, y ,z):
        coords = np.array([lat_lon_boundary[x_y_boundary.index((x[i], y[i]))] for i, val in enumerate(x)])
        x, y = coords[:, 0], coords[:, 1]
        return x, y, 0

    with patch.object(pyproj, 'transform', map_coords) as proj:
        footprint = csm.generate_latlon_footprint(mock_camera, n_points=11, semi_major=10, semi_minor=10)

    assert expected == footprint.GetGeometryCount()

@pytest.mark.parametrize('build_coord_map', ([170, 172, 173, 174, 11],), indirect = True)
@pytest.mark.parametrize('n_points, expected', [(11, 1)])
def test_latlon_not_prime_footprint(build_coord_map, n_points, expected, mock_camera):
    lat_lon_boundary, x_y_boundary = build_coord_map

    def map_coords(arg1, arg2, x, y ,z):
        coords = np.array([lat_lon_boundary[x_y_boundary.index((x[i], y[i]))] for i, val in enumerate(x)])
        x, y = coords[:, 0], coords[:, 1]
        return x, y, 0

    with patch.object(pyproj, 'transform', map_coords) as proj:
        footprint = csm.generate_latlon_footprint(mock_camera, n_points=11, semi_major=10, semi_minor=10)

        assert expected == footprint.GetGeometryCount()

@pytest.mark.parametrize('build_coord_map', ([179, 180, -180, -179, 11],), indirect = True)
@pytest.mark.parametrize('n_points, expected', [(11, 2)])
def test_bodyfixed_prime_footprint(build_coord_map, n_points, expected, mock_camera):
    lat_lon_boundary, x_y_boundary = build_coord_map

    def map_coords(arg1, arg2, x, y ,z):
        try:
            coords = np.array([lat_lon_boundary[x_y_boundary.index((x[i], y[i]))] for i, val in enumerate(x)])
        except:
            coords = np.array([x_y_boundary[lat_lon_boundary.index((np.round(x[i], 2), np.round(y[i],2)))] for i, val in enumerate(x)])
        x, y = coords[:, 0], coords[:, 1]
        return x, y, 0

    with patch.object(pyproj, 'transform', map_coords) as proj:
        footprint = csm.generate_bodyfixed_footprint(mock_camera, n_points=11, semi_major=10, semi_minor=10)

    assert expected == footprint.GetGeometryCount()

@pytest.mark.parametrize('build_coord_map', ([170, 172, 173, 174, 11],), indirect = True)
@pytest.mark.parametrize('n_points, expected', [(11, 1)])
def test_bodyfixed_not_prime_footprint(build_coord_map, n_points, expected, mock_camera):
    lat_lon_boundary, x_y_boundary = build_coord_map

    def map_coords(arg1, arg2, x, y ,z):
        try:
            coords = np.array([lat_lon_boundary[x_y_boundary.index((x[i], y[i]))] for i, val in enumerate(x)])
        except:
            coords = np.array([x_y_boundary[lat_lon_boundary.index((np.round(x[i], 2), np.round(y[i],2)))] for i, val in enumerate(x)])
        x, y = coords[:, 0], coords[:, 1]
        return x, y, 0

    with patch.object(pyproj, 'transform', map_coords) as proj:
        footprint = csm.generate_bodyfixed_footprint(mock_camera, n_points=11, semi_major=10, semi_minor=10)

    assert expected == footprint.GetGeometryCount()
