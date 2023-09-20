# FyTok

## Description

Tokamak integrated modeling and analysis toolkit.

## Software Architecture

Software architecture description

![Image](docs/figures/fytok.svg "FuYun")

## Installation

### Requirements:

### Build dependence

- IMAS data dictionary > 3.38.1 # IMAS DD 源文件
- fytok_data # 装置相关的映射文件 <fytok_data>/mapping

### Develop Build (inplace)

Create IMAS wrapper

```{bash}

export IMAS_DD_PATH= /home/salmon/workspace/data-dictionary
export FYTOK_MAPPING_PATH=/home/salmon/workspace/fytok_data/mapping

cd $IMAS_DD_PATH
patch -s -p0 < <fytok dir>/python/setup_helper/dd_<dd version>.patch


cd <fytok dir>/python
python3 setup.py install_imas_wrapper --prefix=./
```

### Release Build

TODO:...

```{bash}
cd <fytok dir>/python
python3 setup.py bist_
```

## Instructions

## Contribution

## Feature
