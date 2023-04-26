"""
Generate IMAS wrapper  from data dictionary for FyTok

Create by :
    Zhi YU, yuzhi@ipp.ac.cn

Changes:
    2023-04-22: 0.0.1,  ZY, initial version
    2023-04-23:         ZY, add dd_3.38.1.patch DD/include/IDSdef.xml 目录下少量constants定义未生成，已根据源文件补充

TODO:
    - generate wrapper from dd source (xsd)
"""

import saxonche as saxonc
import pathlib
import collections.abc
import subprocess

FYTOK_REV = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

IDS_LIST = []

DD_VERSION = "3.38.1"

DD_PATH = pathlib.Path(f"/fuyun/software/data-dictionary/{DD_VERSION}/dd_{DD_VERSION}")

FY_PATH = pathlib.Path(__file__).parent.parent


def convert_value(proc: saxonc.PySaxonProcessor, v):
    if isinstance(v, str):
        return proc.make_string_value(v)
    elif isinstance(v, int):
        return proc.make_integer_value(v)
    elif isinstance(v, bool):
        return proc.make_boolean_value(v)
    elif isinstance(v, collections.abc.Sequence):
        return proc.make_array([convert_value(proc, d) for d in v])
    elif isinstance(v, collections.abc.Mapping):
        return proc.make_map({k: convert_value(proc, d) for k, d in v.items()})
    else:
        raise TypeError(f"Unsupported type {type(v)}")


def apply_xslt(source_file=None, stylesheet_file=None, target_path="./", **kwargs):

    if stylesheet_file is None:
        stylesheet_file = (FY_PATH/"builder/fy_imas.xsl").as_posix()

    with saxonc.PySaxonProcessor(license=False) as proc:
        xslt_processor = proc.new_xslt30_processor()

        for k, v in kwargs.items():
            xslt_processor.set_parameter(k, convert_value(proc, v))

        executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file)

        executable.transform_to_file(source_file=source_file, output_file=f"{target_path}/ids_list")


if __name__ == "__main__":

    apply_xslt(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
               target_path=(FY_PATH/"python/_imas").as_posix(),
               FYTOK_REV=FYTOK_REV)
