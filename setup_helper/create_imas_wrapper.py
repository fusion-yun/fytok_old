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


def create_imas_warpper(dd_file: pathlib.Path | str,
                        target_path=None,
                        stylesheet_file=None,
                        git_describe=None,
                        **kwargs):

    dd_file = pathlib.Path(dd_file)

    if dd_file.is_dir():
        dd_file = dd_file/"dd_physics_data_dictionary.xsd"

    if not dd_file.exists:
        raise FileExistsError(f"Can not find IMAS data dictionary schema files! {dd_file}")

    dd_git_describe = subprocess.check_output(
        ['git', 'describe', '--always', '--dirty'], cwd=dd_file.parent).strip().decode('utf-8')

    current_dir = pathlib.Path(__file__).parent

    if git_describe is None:
        git_describe = subprocess.check_output(
            ['git', 'describe', '--always', '--dirty'], cwd=current_dir).strip().decode('utf-8')

    if stylesheet_file is None:
        stylesheet_file = current_dir/"fy_dd_wrapper.xsl"

    if not stylesheet_file.exists():
        raise FileExistsError(f"Can not find stylesheet file! {stylesheet_file}")

    if target_path is None:
        target_path = current_dir.parent/"_imas"

    with saxonc.PySaxonProcessor(license=False) as proc:
        xslt_processor = proc.new_xslt30_processor()

        xslt_processor.set_parameter("FY_GIT_DESCRIBE", convert_value(proc, git_describe))
        xslt_processor.set_parameter("DD_GIT_DESCRIBE", convert_value(proc, dd_git_describe))
        xslt_processor.set_parameter("DD_BASE_DIR",     convert_value(proc, dd_file.parent.as_posix()+"/"))

        executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file.as_posix())

        executable.transform_to_file(source_file=dd_file.as_posix(),
                                     output_file=(target_path/"ids_list").as_posix())


if __name__ == "__main__":

    create_imas_warpper(dd_file="/home/salmon/workspace/data-dictionary/dd_physics_data_dictionary.xsd")
