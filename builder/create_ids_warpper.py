import saxonche as saxonc
import pathlib
import pprint
import datetime
import os

import subprocess

FYTOK_REV = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')

IDS_LIST = "(pulse_schedule,distributions,equilibrium)"

DD_VERSION = "3.38.1"

DD_PATH = pathlib.Path(f"/fuyun/software/data-dictionary/{DD_VERSION}/dd_{DD_VERSION}")


stylesheet_file = "/home/salmon/workspace/fytok/builder/fy_ids.xsl"

output_path = pathlib.Path("/home/salmon/workspace/fytok/python/_imas")

print(os.path.relpath(__file__, stylesheet_file))


with saxonc.PySaxonProcessor(license=False) as proc:
    xslt_processor = proc.new_xslt30_processor()
    xslt_processor.set_parameter("FYTOK_REV", proc.make_string_value(FYTOK_REV))
    xslt_processor.set_parameter("IDS_LIST", proc.make_string_value(IDS_LIST))

    # xslt_processor.compile_stylesheet(stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")
    # result = xslt_processor.transform_to_string(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
    # stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")

    executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file)

    result = executable.transform_to_file(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
                                          output_file=(output_path/"ids_list").as_posix())
