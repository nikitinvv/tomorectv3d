from pkg_resources import get_distribution, DistributionNotFound

from tomorectv3d.tomorectv import *
from tomorectv3d.solver import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass