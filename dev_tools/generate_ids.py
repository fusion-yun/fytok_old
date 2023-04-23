import saxonche as saxonc
import pathlib
import pprint
import datetime

ids_name = "(pulse_schedule,distributions)"

DD_VERSION = "3.38.1"

DD_PATH = pathlib.Path(f"/fuyun/software/data-dictionary/{DD_VERSION}/dd_{DD_VERSION}")

stylesheet_path = pathlib.Path("/home/salmon/workspace/fytok/dev_tools")
output_path = pathlib.Path("/home/salmon/workspace/fytok/python/_imas")


with saxonc.PySaxonProcessor(license=False) as proc:
    xslt_processor = proc.new_xslt30_processor()

    xslt_processor.set_parameter("IDS_NAME", proc.make_string_value(ids_name))
    
    xslt_processor.set_parameter("CURRENT_DATETIME", proc.make_string_value(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # xslt_processor.compile_stylesheet(stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")
    # result = xslt_processor.transform_to_string(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
    # stylesheet_file="/home/salmon/workspace/fytok/dev_tools/fy_ids.xsl")
    
    executable = xslt_processor.compile_stylesheet(stylesheet_file=(stylesheet_path/"fy_ids.xsl").as_posix())
    
    result = executable.transform_to_file(source_file=(DD_PATH/"include/IDSDef.xml").as_posix(),
                                          output_file=(output_path/"__init__.py").as_posix())
