__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '0.0.0'


from spdm.util.logger import logger
from spdm.data.File import File

logger.info("Regisiter Plugins: File [mds,geqdsk]")

File.associtaion.update({
    "file.mds": ".data.db.MDSplus#MDSplusDocument",
    "file.mdsplus": ".data.db.MDSplus#MDSplusDocument",
    "file.gfile": ".data.file.PluginGEQdsk",
    "file.geqdsk": ".data.file.PluginGEQdsk",
})
