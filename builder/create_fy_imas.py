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
    
    DD_VERSION = "3.38.1"
    
    DD_BASE_DIR = pathlib.Path(f"/home/salmon/workspace/data-dictionary")   
        
    DD_GIT_DESCRIBE = subprocess.check_output(['git', 'describe', '--always', '--dirty'], cwd=DD_BASE_DIR).strip().decode('utf-8')

    FY_BASE_DIR = pathlib.Path(__file__).parent.parent

    FY_GIT_DESCRIBE = subprocess.check_output(['git', 'describe', '--always', '--dirty'], cwd=FY_BASE_DIR).strip().decode('utf-8')
    
    # apply_xslt(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
    #            target_path=(FY_PATH/"python/_imas").as_posix(),
    #            FYTOK_REV=FYTOK_REV)
    
    apply_xslt(stylesheet_file  =(FY_BASE_DIR/"builder/fy_imas_xsd.xsl").as_posix(),
               target_path      =(FY_BASE_DIR/"python/_imas").as_posix(),
               source_file      =(DD_BASE_DIR/"dd_physics_data_dictionary.xsd").as_posix(),
               FY_GIT_DESCRIBE  =FY_GIT_DESCRIBE,
               DD_GIT_DESCRIBE  =DD_GIT_DESCRIBE,
               DD_BASE_DIR      =DD_BASE_DIR.as_posix()+"/")
