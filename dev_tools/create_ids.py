import argparse
import pathlib

ids_templ = """
# This is file is generated from template
from fytok.IDS import IDS

class {clsname}(IDS):
    r\"\"\"{doc_string}
        .. note:: {clsname} is an ids
    \"\"\"
    IDS="{ids}"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
"""


def create_ids(*, input_path, output_path=None, **kwargs):
    prefix = pathlib.Path(output_path or "./")
    with open(input_path, "r") as reader:
        for line in reader:
            k, desc = line.split("::")
            k = k.strip()
            desc = desc.strip()
            clsname = ''.join(x.capitalize() or '_' for x in k.split('_'))
            print(ids_templ.format(clsname=clsname, ids=k, doc_string=desc))
            print(prefix/f"{clsname}.py")
            with open(prefix/f"{clsname}.py", "w") as ofile:
                ofile.write(ids_templ.format(clsname=clsname, ids=k, doc_string=desc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate IDS　handler.')
    parser.add_argument('-i', type=str, help='an integer for the accumulator')
    parser.add_argument('-o',  type=str,  help='sum the integers (default: find the max)')
    args = parser.parse_args()
    create_ids(input_path=args.i, output_path=args.o)
