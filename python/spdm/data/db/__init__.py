__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '0.0.0'


from spdm.util.logger import logger
from spdm.data.DataBase import DataBase

logger.info("Regisiter DataBase Plugins [mds, imas]")

DataBase.associtaion.update({
    "db.imas": ".data.db.IMAS#IMASDocument",
    "db.mds": ".data.db.MDSplus#MDSplusCollection",
    "db.mdsplus": ".data.db.MDSplus#MDSplusCollection",
})
