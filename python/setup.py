# setup.py
#
# Author: Zhi YU
# Created Time: 2015-12-28 21:58:47
#
# NOTE:
#  -  this file is used to install the fytok package
# Usage:
#  生成 IMAS wrapper
#  -  python3 setup.py install_imas_wrapper --prefix=.

import collections.abc
import os
import pathlib
import pprint
import shutil
import subprocess

from setuptools import Command, find_namespace_packages, setup
from setuptools.command.build_py import build_py

# Get version from git, '--abbrev=0'
version = (
    subprocess.check_output(["git", "describe", "--always", "--dirty"])
    .strip()
    .decode("utf-8")
)

fy_git_describe = (
    subprocess.check_output(["git", "describe", "--always", "--dirty"])
    .strip()
    .decode("utf-8")
)

SRC_ROOT = pathlib.Path(__file__).parent

SETUP_HELPER_DIR = SRC_ROOT / "_setup_helper"

# Get the long description from the README file
with open("../README.md") as f:
    long_description = f.read()

with open("../LICENSE.txt") as f:
    license = f.read()

# Get the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()


def convert_value(proc, v):
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


def fetch_url(url, tmp_dir=None):
    import json
    import urllib.parse
    import urllib.request

    url = urllib.parse.urlparse(url)

    if url.scheme == "https":
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler())
    elif url.scheme == "http":
        opener = urllib.request.build_opener(urllib.request.HTTPHandler())
    else:
        raise ValueError(f"Unsupported scheme {url.scheme}!")

    req = urllib.request.Request(
        url.geturl(),
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0"
        },
    )

    with opener.open(req) as f:
        return json.load(f)


def cerate_symlink(prefix, dd_version):
    cwd = os.getcwd()

    os.chdir(prefix)

    if os.path.exists(dd_version):
        if os.path.islink("lastest"):
            os.unlink("lastest")
        elif os.path.exists("lastest"):
            shutil.rmtree("lastest")
            # raise FileExistsError("lastest is not a symlink! Please remove it manually!")
        print(f"cp -r  {dd_version} lastest")
        # os.symlink(dd_version, "lastest")
        shutil.copytree(dd_version, "lastest")
    else:
        raise FileExistsError(f"Can not find IMAS wrapper! {prefix/dd_version}")

    os.chdir(cwd)


def create_imas_warpper(
    target_path,
    dd_path: str,
    xsl_file: str = None,
    xsl_schema_file: str = None,
    symlink_as_lastest=True,
):
    """Create IMAS warpper for python"""

    # 用logger输出log信息

    dd_path = pathlib.Path(dd_path)

    if dd_path.suffix != ".xsd":
        dd_path = dd_path / "dd_physics_data_dictionary.xsd"

    if not dd_path.exists():
        raise FileExistsError(f"Can not find IMAS data dictionary! {dd_path}")

    dd_git_describe = (
        subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], cwd=dd_path.parent
        )
        .strip()
        .decode("utf-8")
    )

    if xsl_file is None:
        xsl_file = SETUP_HELPER_DIR / "fy_imas_python.xsl"
    else:
        xsl_file = pathlib.Path(xsl_file)

    if xsl_schema_file is None:
        xsl_schema_file = SETUP_HELPER_DIR / "fy_imas_schema.xsl"
    else:
        xsl_schema_file = pathlib.Path(xsl_schema_file)

    if xsl_file is None or not pathlib.Path(xsl_file).exists():
        raise FileExistsError(f"Can not find stylesheet file! {xsl_file}")

    dd_version = "v" + dd_git_describe.replace(".", "_").replace("-", "_")

    target_path = pathlib.Path(target_path)

    import saxonche as saxonc

    with saxonc.PySaxonProcessor(license=False) as proc:
        xslt_processor = proc.new_xslt30_processor()

        xslt_processor.set_parameter(
            "FY_GIT_DESCRIBE", convert_value(proc, fy_git_describe)
        )
        xslt_processor.set_parameter(
            "DD_GIT_DESCRIBE", convert_value(proc, dd_git_describe)
        )
        xslt_processor.set_parameter(
            "DD_BASE_DIR", convert_value(proc, dd_path.parent.as_posix() + "/")
        )

        print(f"Create IMAS Python warpper: {target_path}/_imas/{dd_version}")
        xslt_processor.compile_stylesheet(
            stylesheet_file=xsl_file.as_posix()
        ).transform_to_file(
            source_file=dd_path.as_posix(),
            output_file=(
                target_path / "_imas" / dd_version / "._physics_data_dictionary.txt"
            ).as_posix(),
        )

        # print(f"Create IMAS Schema: {target_path}/_schema/{dd_version}")
        # xslt_processor\
        #     .compile_stylesheet(stylesheet_file=xsl_schema_file.as_posix())\
        #     .transform_to_file(source_file=dd_path.as_posix(),
        #                        output_file=(target_path/"_schema"/dd_version/"imas_physics_data_dictionary.yaml").as_posix())

    if symlink_as_lastest:
        cerate_symlink(target_path / "_imas", dd_version)
        # cerate_link(target_path/"_schema", dd_version)


def copy_data_mapping(target_path, mapping_path: str):
    """Copy data mapping for IMAS warpper"""

    # 用logger输出log信息
    print(f"Copy device data mapping for IMAS warpper to {target_path}")

    target_path = pathlib.Path(target_path)

    mapping_path = pathlib.Path(mapping_path) / "mapping"

    if mapping_path.exists():
        shutil.copytree(mapping_path, target_path / "_mapping", dirs_exist_ok=True)
    else:
        print(f"Can not find data mapping! {mapping_path}")


class InstallIMASWrapper(Command):
    description = "Install IMAS Wrapper"
    user_options = [
        ("prefix=", None, "Prefix for IMAS wrapper"),
        ("as-lastest=", None, "Symlink as lastest IMAS wrapper"),
        ("dd-path=", None, "Path of IMAS data dictionary"),
        ("mapping-path=", None, "Path of IMAS data dictionary"),
        ("stylesheet-file=", None, "Stylesheet file for generating IMAS wrapper"),
    ]

    def initialize_options(self):
        self.prefix = SRC_ROOT
        self.as_lastest = True
        self.dd_path = os.environ.get("IMAS_PREFIX", None) or os.environ.get(
            "IMAS_DD_PATH", "/home/salmon/workspace/data-dictionary"
        )
        self.mapping_path = os.environ.get(
            "FYTOK_MAPPING_PATH", "/home/salmon/workspace/fytok_data"
        )
        self.stylesheet_file = None

    def finalize_options(self):
        pass

    def run(self):
        target_path = pathlib.Path(self.prefix) / "fytok"

        create_imas_warpper(
            target_path=(target_path).as_posix(),
            dd_path=self.dd_path,
            xsl_file=self.stylesheet_file,
            symlink_as_lastest=self.as_lastest,
        )

        copy_data_mapping(
            target_path=(target_path).as_posix(), mapping_path=self.mapping_path
        )


class BuildPyCommand(build_py):
    description = "Install __doc__,__version, and IMAS Wrapper"
    user_options = build_py.user_options + [
        ("dd-path=", None, "Path of IMAS data dictionary"),
        ("mapping-path=", None, "Path of IMAS mapping files"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.dd_path = os.environ.get("IMAS_PREFIX", None) or os.environ.get(
            "IMAS_DD_PATH", "/home/salmon/workspace/data-dictionary"
        )

        self.mapping_path = os.environ.get(
            "FYTOK_MAPPING_PATH", "/home/salmon/workspace/fytok_data"
        )

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        build_dir = pathlib.Path(self.build_lib) / f"{self.distribution.get_name()}"

        build_dir.mkdir(parents=True, exist_ok=True)

        super().run()

        with open(build_dir / "__version__.py", "w") as f:
            f.write(f'__version__ = "{self.distribution.get_version()}"')

        if not (build_dir / "__doc__.py").exists():
            with open(build_dir / "__doc__.py", "w") as f:
                f.write(f'"""\n{self.distribution.get_long_description()}\n"""')

        create_imas_warpper(
            target_path=(build_dir / "_imas").as_posix(),
            dd_path=self.dd_path,
            symlink_as_lastest=True,
        )

        copy_data_mapping(
            target_path=(build_dir / "_mapping").as_posix(),
            mapping_path=self.mapping_path,
        )


# Setup the package
setup(
    name="fytok",
    version=version,
    description=f"Fusion Tokamak Simulation Toolkit",
    long_description=long_description,
    url="http://fytok.github.io",
    author="Zhi YU",
    author_email="yuzhi@ipp.ac.cn",
    license=license,
    cmdclass={
        "build_py": BuildPyCommand,
        "install_imas_wrapper": InstallIMASWrapper,
    },
    packages=find_namespace_packages(
        "python",
        include=["fytok", "fytok.*", "_imas", "_imas.*", "_mapping", "_mapping.*"],
    ),  # 指定需要安装的包
    # requires=requirements,              # 项目运行依赖的第三方包
    setup_requires=["saxonche"],  # 项目构建依赖的第三方包
    classifiers=[
        "Development Status :: 0 - Beta",
        "Intended Audience :: Plasma Physicists",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="plasma physics",  # 关键字列表
    python_requires=">=3.10, <4",  # Python版本要求
)
