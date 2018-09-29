from csmapi import csmapi
import warnings

# Register the usgscam plugin with the csmapi
from distutils import sysconfig
import ctypes

lib = ctypes.CDLL(os.path.abspath(os.path.join(sysconfig.get_python_lib(), '../../libusgscsm.so')))
if not lib:
    warnings.warn('Unable to load usgscsm shared library')
