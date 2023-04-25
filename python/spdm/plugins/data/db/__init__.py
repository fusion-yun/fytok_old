__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '0.0.0'


from spdm.utils.logger import logger
from .SpObject import SpObject


SpObject.association.update({
    "db.imas": ".data.db.IMAS#IMASDocument",
})
