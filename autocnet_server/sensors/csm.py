import datetime
import json
import os

from csmapi import csmapi
import jinja2
import requests

import numpy as np
import pyproj
import gdal
from gdal import ogr
import pvl

from plio.utils.utils import find_in_dict
from plio.io.io_json import NumpyEncoder
from plio.spatial.footprint import generate_gcps

def data_from_cube(header):
    """
    Take an ISIS Cube header and normalize back to PVL keywords.
    """
    instrument_name = 'CONTEXT CAMERA'
    data = pvl.PVLModule([('START_TIME', find_in_dict(header, 'StartTime')),
                          ('SPACECRAFT_NAME', find_in_dict(header, 'SpacecraftName').upper()),
                          ('INSTRUMENT_NAME', instrument_name),
                          ('SAMPLING_FACTOR', find_in_dict(header, 'SpatialSumming')),
                          ('SAMPLE_FIRST_PIXEL', find_in_dict(header, 'SampleFirstPixel')),
                          ('TARGET_NAME', find_in_dict(header, 'TargetName').upper()),
                          ('LINE_EXPOSURE_DURATION', find_in_dict(header, 'LineExposureDuration')),
                          ('SPACECRAFT_CLOCK_START_COUNT', find_in_dict(header, 'SpacecraftClockCount')),
                          ('IMAGE', {'LINES':find_in_dict(header, 'Lines'),
                                    'LINE_SAMPLES':find_in_dict(header, 'Samples')})])

    return data

def create_camera(obj, url='http://pfeffer.wr.usgs.gov/v1/pds/',
                 plugin_name='USGS_ASTRO_LINE_SCANNER_PLUGIN',
                 model_name='USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'):
    data = data_from_cube(obj.metadata)

    data_serialized = {'label': pvl.dumps(data).decode()}
    r = requests.post(url, json=data_serialized).json()
    r['IKCODE'] = -1
    # Get the ISD back and instantiate a local ISD for the image
    isd = csmapi.Isd.loads(r)

    # Create the plugin and camera as usual
    plugin = csmapi.Plugin.findPlugin(plugin_name)
    if plugin.canModelBeConstructedFromISD(isd, model_name):
        return plugin.constructModelFromISD(isd, model_name)

def generate_boundary(camera, nnodes=5, n_points=10):
    '''
    Generates a bounding box given a camera model

    Parameters
    ----------
    camera : object
             csmapi generated camera model

    nnodes : int
             Not sure

    n_points : int
               Number of points to generate between the corners of the bounding
               box per side.

    Returns
    -------
    boundary : lst
               List of full bounding box
    '''
    isize = camera.getImageSize()
    x = np.linspace(0, isize.samp, n_points)
    y = np.linspace(0, isize.line, n_points)
    boundary = [(i,0.) for i in x] + [(isize.samp, i) for i in y[1:]] +\
               [(i, isize.line) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]

    return boundary

def generate_latlon_boundary(camera, nnodes=5, semi_major=3396190, semi_minor=3376200, n_points=10):
    '''
    Generates a latlon bounding box given a camera model

    Parameters
    ----------
    camera : object
             csmapi generated camera model

    nnodes : int
             Not sure

    semi_major : int
                 Semimajor axis of the target body

    semi_minor : int
                 Semiminor axis of the target body

    n_points : int
               Number of points to generate between the corners of the bounding
               box per side.

    Returns
    -------
    lons : lst
           List of longitude values

    lats : lst
           List of latitude values

    alts : lst
           List of altitude values
    '''
    boundary = generate_boundary(camera, nnodes=nnodes, n_points = n_points)

    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    gnds = np.empty((len(boundary), 3))

    for i, b in enumerate(boundary):
        gnd = camera.imageToGround(csmapi.ImageCoord(*b), 0)
        gnds[i] = [gnd.x, gnd.y, gnd.z]

    lons, lats, alts = pyproj.transform(ecef, lla, gnds[:,0], gnds[:,1], gnds[:,2])
    return lons, lats, alts

def generate_gcps(camera, nnodes=5, semi_major=3396190, semi_minor=3376200, n_points=10):
    '''
    Generates an area of ground control points formated as:
    <GCP Id="" Info="" Pixel="" Line="" X="" Y="" Z="" /> per record

    Parameters
    ----------
    camera : object
             csmapi generated camera model

    nnodes : int
             Not sure

    semi_major : int
                 Semimajor axis of the target body

    semi_minor : int
                 Semiminor axis of the target body

    n_points : int
               Number of points to generate between the corners of the bounding
               box per side.

    Returns
    -------
    gcps : lst
           List of all gcp records generated
    '''
    lons, lats, alts = generate_latlon_boundary(camera, nnodes=nnodes,
                                                semi_major=semi_major,
                                                semi_minor=semi_minor,
                                                n_points=n_points)

    lla = np.vstack((lons, lats, alts)).T

    tr = zip(boundary, lla)

    gcps = []
    for i, t in enumerate(tr):
        l = '<GCP Id="{}" Info="{}" Pixel="{}" Line="{}" X="{}" Y="{}" Z="{}" />'.format(i, i, t[0][1], t[0][0], t[1][0], t[1][1], t[1][2])
        gcps.append(l)

    return gcps

def generate_latlon_footprint(camera, nnodes=5, semi_major=3396190, semi_minor=3376200, n_points=10):
    '''
    Generates a latlon footprint from a csmapi generated camera model

    Parameters
    ----------
    camera : object
             csmapi generated camera model

    nnodes : int
             Not sure

    semi_major : int
                 Semimajor axis of the target body

    semi_minor : int
                 Semiminor axis of the target body

    n_points : int
               Number of points to generate between the corners of the bounding
               box per side.

    Returns
    -------
    : object
      ogr multipolygon containing between one and two polygons
    '''
    lons, lats, _ = generate_latlon_boundary(camera, nnodes=nnodes,
                                                semi_major=semi_major,
                                                semi_minor=semi_minor,
                                                n_points=n_points)

    ll_coords = [*zip(((lons + 180) % 360), lats)]

    ring = ogr.Geometry(ogr.wkbLinearRing)
    wrap_ring = ogr.Geometry(ogr.wkbLinearRing)
    poly = ogr.Geometry(ogr.wkbPolygon)
    wrap_poly = ogr.Geometry(ogr.wkbPolygon)
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)

    current_ring = ring
    switch_point = None
    previous_point = None

    for coord in ll_coords:

        if previous_point:
            coord_diff = previous_point[0] - coord[0]

            if coord_diff > 0 and np.isclose(previous_point[0], 360, rtol = 1e-03) and \
                                  np.isclose(coord[0], 0, atol=1e0, rtol=1e-01):
                slope, b = compute_line(previous_point, coord)
                current_ring.AddPoint(360 - 180, (slope*360 + b))
                current_ring = wrap_ring
                switch_point = 0 - 180, (slope*0 + b)
                current_ring.AddPoint(*switch_point)

            elif coord_diff < 0 and np.isclose(previous_point[0], 0, atol=1e0, rtol=1e-01) and \
                                    np.isclose(coord[0], 360, rtol = 1e-03):
                slope, b = compute_line(previous_point, coord)
                current_ring.AddPoint(0 - 180, (slope*0 + b))
                current_ring.AddPoint(*switch_point)
                current_ring = ring
                current_ring.AddPoint(360 - 180, (slope*360 + b))

        lat, lon = coord
        current_ring.AddPoint(lat - 180, lon)
        previous_point = coord

    poly.AddGeometry(ring)
    wrap_poly.AddGeometry(wrap_ring)

    if not wrap_poly.IsEmpty():
        multipoly.AddGeometry(wrap_poly)

    if not poly.IsEmpty():
        multipoly.AddGeometry(poly)

    return multipoly

def generate_bodyfixed_footprint(camera, nnodes=5, n_points=10):
    '''
    Generates a bodyfixed footprint from a csmapi generated camera model

    Parameters
    ----------
    camera : object
             csmapi generated camera model

    nnodes : int
             Not sure

    n_points : int
               Number of points to generate between the corners of the bounding
               box per side.

    Returns
    -------
    : object
      ogr polygon
    '''
    boundary = generate_boundary(camera, nnodes=nnodes, n_points=n_points)

    ring = ogr.Geometry(ogr.wkbLinearRing)

    for i in boundary:
        gnd = camera.imageToGround(*i, 0)
        ring.AddPoint(gnd[0], gnd[1], gnd[2])

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def compute_line(point1, point2):
    '''
    Computes the slope and y-intercept between two points

    Parameters
    ----------
    point1 : tuple
             Tuple of x, y coord

    point2 : tuple
             Tuple of x, y coord

    Returns
    -------
    slope : float
            Slope of the line

    b : float
        y-intercept
    '''
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point2[1] - (slope*point2[0])

    return slope, b

def warped_vrt(camera, raster_size, fpath, outpath=None, no_data_value=0):
    gcps = generate_gcps(camera)
    xsize, ysize = raster_size

    if outpath is None:
        outpath = os.path.dirname(fpath)
    outname = os.path.splitext(os.path.basename(fpath))[0] + '.vrt'
    outname = os.path.join(outpath, outname)

    xsize, ysize = raster_size
    vrt = r'''<VRTDataset rasterXSize="{{ xsize }}" rasterYSize="{{ ysize }}">
     <Metadata/>
     <GCPList Projection="{{ proj }}">
     {% for gcp in gcps -%}
       {{gcp}}
     {% endfor -%}
    </GCPList>
     <VRTRasterBand dataType="Float32" band="1">
       <NoDataValue>{{ no_data_value }}</NoDataValue>
       <Metadata/>
       <ColorInterp>Gray</ColorInterp>
       <SimpleSource>
         <SourceFilename relativeToVRT="0">{{ fpath }}</SourceFilename>
         <SourceBand>1</SourceBand>
         <SourceProperties rasterXSize="{{ xsize }}" rasterYSize="{{ ysize }}"
    DataType="Float32" BlockXSize="512" BlockYSize="512"/>
         <SrcRect xOff="0" yOff="0" xSize="{{ xsize }}" ySize="{{ ysize }}"/>
         <DstRect xOff="0" yOff="0" xSize="{{ xsize }}" ySize="{{ ysize }}"/>
       </SimpleSource>
     </VRTRasterBand>
    </VRTDataset>'''

    context = {'xsize':xsize, 'ysize':ysize,
               'gcps':gcps,
               'proj':'+proj=longlat +a=3396190 +b=3376200 +no_defs',
               'fpath':fpath,
               'no_data_value':no_data_value}
    template = jinja2.Template(vrt)
    tmp = template.render(context)
    warp_options = gdal.WarpOptions(format='VRT', dstNodata=0)
    gdal.Warp(outname, tmp, options=warp_options)
