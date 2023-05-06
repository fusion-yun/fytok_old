# setup.py
#
# Author: Zhi YU
# Created Time: 2015-12-28 21:58:47
#

import collections.abc
import os
import pathlib
import pprint
import subprocess
from setuptools.command.build_py import build_py
from setuptools import Command

from setuptools import find_namespace_packages, setup

git_describe = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')
source_dir = pathlib.Path(__file__).parent

# Get the long description from the README file
with open('../README.md') as f:
    long_description = f.read()

# Get the version from git or the VERSION file
# with open('../VERSION') as f:
#     version = f.read().strip()
version = git_describe

# Get the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_dev.txt') as f:
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


def create_imas_warpper(dd_path: str,
                        target_path=None,
                        stylesheet_file=None,
                        git_describe=None):
    if dd_path is None:
        dd_path = os.environ.get("IMAS_PREFIX", None) or os.environ.get("IMAS_DD_PATH", None)

    if dd_path is None:
        dd_path = "/home/salmon/workspace/data-dictionary"

    if target_path is None:
        raise ValueError("target_path is not specified!")

    if stylesheet_file is None:
        stylesheet_file = (pathlib.Path(__file__).parent/"_build_helper/fy_imas_xsd.xsl")

    stylesheet_file = pathlib.Path(stylesheet_file)
    if not stylesheet_file.exists():
        raise FileExistsError(f"Can not find stylesheet file! {stylesheet_file}")

    dd_path = pathlib.Path(dd_path)
    dd_file = dd_path/"dd_physics_data_dictionary.xsd"
    if not dd_file.exists():
        raise FileExistsError(f"Can not find IMAS data dictionary schema files! {dd_file}")
    else:
        dd_git_describe = subprocess.check_output(
            ['git', 'describe', '--always', '--dirty'], cwd=dd_path).strip().decode('utf-8')

    target_path = pathlib.Path(target_path)

    import saxonche as saxonc

    with saxonc.PySaxonProcessor(license=False) as proc:
        xslt_processor = proc.new_xslt30_processor()

        xslt_processor.set_parameter("FY_GIT_DESCRIBE", convert_value(proc, git_describe))
        xslt_processor.set_parameter("DD_GIT_DESCRIBE", convert_value(proc, dd_git_describe))
        xslt_processor.set_parameter("DD_BASE_DIR",     convert_value(proc, dd_path.as_posix()+"/"))

        executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file.as_posix())

        executable.transform_to_file(source_file=dd_file.as_posix(), output_file=(target_path/"ids_list").as_posix())


class BuildIMASWrapperCommand(Command):
    description = 'Build IMAS Wrapper'
    user_options = [
        ('dd-path=', None, 'Path of IMAS data dictionary'),
        ('target-path=', None, 'Path of IMAS wrapper'),
        ('stylesheet-file=', None, 'Stylesheet file for generating IMAS wrapper'),
    ]

    def initialize_options(self):
        self.dd_path = None
        self.target_path = None
        self.stylesheet_file = None

    def finalize_options(self):
        pass

    def run(self):

        if self.target_path is None:
            self.target_path = pathlib.Path(__file__).parent/f"{self.distribution.get_name()}/_imas"

        create_imas_warpper(dd_path=self.dd_path,
                            target_path=self.target_path,
                            stylesheet_file=self.stylesheet_file,
                            git_describe=git_describe)


class BuildPyCommand(build_py):
    description = 'Build __doc__,__version, and IMAS Wrapper'
    user_options = [
        ('dd-path=', None, 'Path of IMAS data dictionary'),
        ('target-path=', None, 'Path of IMAS wrapper'),
        ('stylesheet-file=', None, 'Stylesheet file for generating IMAS wrapper'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.dd_path = None
        self.target_path = None
        self.stylesheet_file = None

    def finalize_options(self):
        super().finalize_options()
        pass

    def run(self):

        build_py.run(self)

        build_path = pathlib.Path(self.build_lib)

        with open(build_path/'fytok/__version__.py', 'w') as f:
            f.write(f"__version__ = \"{self.distribution.get_version()}\"")

        if not (source_dir/f"{self.distribution.get_version()}/__doc__.py").exists():
            with open(build_path/'fytok/__doc__.py', 'w') as f:
                f.write(f'"""\n{self.distribution.get_long_description()}\n"""')

        if self.target_path is None:
            self.target_path = pathlib.Path(self.build_lib)/f"{self.distribution.get_name()}/_imas"

        create_imas_warpper(dd_path=self.dd_path,
                            target_path=self.target_path,
                            stylesheet_file=self.stylesheet_file,
                            git_describe=git_describe)


# Setup the package
setup(
    name='fytok',
    version=version,
    description=f'Fusion Tokamak Simulation Toolkit {version}',
    long_description=long_description,
    url='http://fytok.github.io',
    author='Zhi YU',
    author_email='yuzhi@ipp.ac.cn',
    license='MIT',

    cmdclass={
        'build_py': BuildPyCommand,
        'build_imas_wrapper': BuildIMASWrapperCommand,
    },

    packages=find_namespace_packages(include=["fytok", "fytok.*", "_imas", "_imas.*"]),  # 指定需要安装的包

    # requires=requirements,              # 项目运行依赖的第三方包

    setup_requires=['saxonche'],        # 项目构建依赖的第三方包

    classifiers=[
        'Development Status :: 0 - Beta',
        'Intended Audience :: Plasma Physicists',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='plasma physics',  # 关键字列表
    python_requires='>=3.10, <4',  # Python版本要求


)
