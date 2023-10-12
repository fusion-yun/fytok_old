import importlib.util
import pathlib
import subprocess

from setuptools import find_namespace_packages
from distutils.core import setup


# Get version from git, '--abbrev=0'
version = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')

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


# Setup the package
setup(
    name="fytok",
    version=version,
    description=f"FuYun Tokamak  Toolkit",
    long_description=long_description,
    url="http://fytok.github.io",
    author="Zhi YU",
    author_email="yuzhi@ipp.ac.cn",
    license=license,

    packages=find_namespace_packages(
        "python", exclude=["*._*", "*.todo", "*.todo.*", "*obsolete", "*.tests"]),  # 指定需要安装的包

    package_dir={"": "python"},  # 指定包的root目录


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
