"""
Generate IMAS wrapper for FyTok from data dictionary

Create by :
    Zhi YU, yuzhi@ipp.ac.cn

Changes:
    2023-04-22: 0.0.1,  ZY, initial version
    2023-04-23:         ZY, add dd_3.38.1.patch DD/include/IDSdef.xml 目录下少量constants定义未生成，已根据源文件补充

"""

import saxonche as saxonc
import pathlib
import pprint
import datetime
import os

import subprocess

FYTOK_REV = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

IDS_LIST = []

DD_VERSION = "3.38.1"

DD_PATH = pathlib.Path(f"/fuyun/software/data-dictionary/{DD_VERSION}/dd_{DD_VERSION}")


stylesheet_file = "/home/salmon/workspace/fytok/builder/fy_ids.xsl"

output_path = pathlib.Path("/home/salmon/workspace/fytok/python/_imas")

print(os.path.relpath(__file__, stylesheet_file))


with saxonc.PySaxonProcessor(license=False) as proc:
    xslt_processor = proc.new_xslt30_processor()
    xslt_processor.set_parameter("FYTOK_REV", proc.make_string_value(FYTOK_REV))
    xslt_processor.set_parameter("IDS_LIST", proc.make_array([*map(proc.make_string_value, IDS_LIST)]))

    # xslt_processor.compile_stylesheet(stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")
    # result = xslt_processor.transform_to_string(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
    # stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")

    executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file)

    result = executable.transform_to_file(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
                                          output_file=(output_path/"ids_list").as_posix())
