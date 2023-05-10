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
from setuptools.command.install import install
from setuptools import Command

from setuptools import find_namespace_packages, setup

# Get version from git
version = subprocess.check_output(['git', 'describe', '--abbrev=0']).strip().decode('utf-8')

fy_git_describe = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

source_dir = pathlib.Path(__file__).parent

# Get the long description from the README file
with open('../README.md') as f:
    long_description = f.read()


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


def create_imas_warpper(target_path, dd_path: str, stylesheet_file: str = None,  symlink_as_lastest=True):

    dd_path = pathlib.Path(dd_path)

    if dd_path.suffix != ".xsd":
        dd_path = dd_path/"dd_physics_data_dictionary.xsd"

    if not dd_path.exists():
        raise FileExistsError(f"Can not find IMAS data dictionary! {dd_path}")

    dd_git_describe = subprocess.check_output(
        ['git', 'describe', '--always', '--dirty'], cwd=dd_path.parent).strip().decode('utf-8')

    if stylesheet_file is None:
        stylesheet_file = pathlib.Path(__file__).parent/"_build_helper/fy_imas_xsd.xsl"
    else:
        stylesheet_file = pathlib.Path(stylesheet_file)

    if stylesheet_file is None or not pathlib.Path(stylesheet_file).exists():
        raise FileExistsError(f"Can not find stylesheet file! {stylesheet_file}")

    dd_dir = "v"+dd_git_describe.replace('.', '_').replace('-', '_')

    target_path = pathlib.Path(target_path)

    import saxonche as saxonc

    with saxonc.PySaxonProcessor(license=False) as proc:
        xslt_processor = proc.new_xslt30_processor()

        xslt_processor.set_parameter("FY_GIT_DESCRIBE", convert_value(proc, fy_git_describe))
        xslt_processor.set_parameter("DD_GIT_DESCRIBE", convert_value(proc, dd_git_describe))
        xslt_processor.set_parameter("DD_BASE_DIR",     convert_value(proc, dd_path.parent.as_posix()+"/"))

        executable = xslt_processor.compile_stylesheet(stylesheet_file=stylesheet_file.as_posix())

        executable.transform_to_file(source_file=dd_path.as_posix(),
                                     output_file=(target_path/dd_dir/"ids_list").as_posix())

    os.chdir(target_path)

    if not symlink_as_lastest:
        pass
    elif os.path.exists(dd_dir):
        if os.path.islink("lastest"):
            os.unlink("lastest")
        elif os.path.exists("lastest"):
            raise FileExistsError("lastest is not a symlink! Please remove it manually!")
        os.symlink(dd_dir, "lastest")
    else:
        raise FileExistsError(f"Can not find IMAS wrapper! {target_path/dd_dir}")


class InstallIMASWrapper(Command):
    description = 'Install IMAS Wrapper'
    user_options = [
        ('prefix=', None, "Prefix for IMAS wrapper"),
        ('as-lastest=', None, 'Symlink as lastest IMAS wrapper'),
        ('dd=', None, 'Path of IMAS data dictionary'),
        ('stylesheet-file=', None, 'Stylesheet file for generating IMAS wrapper'),
    ]

    def initialize_options(self):
        self.prefix = pathlib.Path(__file__).parent
        self.as_lastest = False
        self.dd = os.environ.get("IMAS_PREFIX", None) or\
            os.environ.get("IMAS_DD_PATH", "/home/salmon/workspace/data-dictionary")
        self.stylesheet_file = None

    def finalize_options(self):
        pass

    def run(self):

        target_path = pathlib.Path(self.prefix) / "fytok/_imas"

        create_imas_warpper(target_path=target_path.as_posix(),
                            dd_path=self.dd,
                            stylesheet_file=self.stylesheet_file,
                            symlink_as_lastest=self.as_lastest)


class InstallCommand(install):
    description = 'Install __doc__,__version, and IMAS Wrapper'
    user_options = install.user_options + [
        ('dd=', None, 'Path of IMAS data dictionary'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.dd = os.environ.get("IMAS_PREFIX", None) or\
            os.environ.get("IMAS_DD_PATH", "/home/salmon/workspace/data-dictionary")

    def finalize_options(self):
        super().finalize_options()

    def run(self):

        super().run()

        install_dir = pathlib.Path(self.install_lib)/f"{self.distribution.get_name()}"

        with open(install_dir/'__version__.py', 'w') as f:
            f.write(f"__version__ = \"{self.distribution.get_version()}\"")

        if not (source_dir/f"{self.distribution.get_version()}/__doc__.py").exists():
            with open(install_dir/'__doc__.py', 'w') as f:
                f.write(f'"""\n{self.distribution.get_long_description()}\n"""')

        create_imas_warpper(target_path=(install_dir/"_imas").as_posix(),
                            dd_path=self.dd,
                            symlink_as_lastest=True)


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
        'install': InstallCommand,
        'install_imas_wrapper': InstallIMASWrapper,
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
