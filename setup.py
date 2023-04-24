from setuptools import setup

try:
    import subprocess
    version = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')
except:
    version = "_unknown_"


setup(name="fytok",
      version=version,
      author="Zhi YU"
      author_email="yuzhi@ipp.ac.cn",
      url="",
      install_requires=["spdm"],
      setup_requires=["setuptools", 'saxonche'],
      packages=["fytok"],
      classifiers=[
           "Development Status :: 0 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: Other/Proprietary License",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Physics",
      ]
      )
