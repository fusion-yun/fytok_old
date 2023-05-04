import lxml.etree as etree
import pathlib
import pprint
import datetime

DD_VERSION = "3.38.1"
DD_PATH = pathlib.Path("/fuyun/software/data-dictionary/{DD_VERSION}/dd_{DD_VERSION}")


class FyIDSGenerator(object):

    def __init__(self,):
        self._dom = etree.parse(DD_PATH/"include/IDSDef.xml")
        self._xslt = etree.parse("./fy_ids.xsl")
        self._transform = etree.XSLT(self._xslt)

    def write(self, ids_name, output_path=".", force=False):
        output_path = pathlib.Path(pathlib.Path(output_path)/f"{ids_name.upper()}.py")

        if force:
            output_path.parent.mkdir(parents=True)

        if not force and output_path.exists():
            raise FileExistsError(output_path)

        try:
            output = self._transform(self._dom,
                                     IDS_NAME=ids_name,
                                     CURRENT_DATATIME=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except:
            for error in self._transform.error_log:
                print(error.message, error.line)

        output.write_output(output_path.as_posix())
