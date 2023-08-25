# FyTok

## Description
  
  Tokamak integrated modeling and analysis toolkit.

## Software Architecture
Software architecture description

![Image](docs/figures/fytok.svg "FuYun")

## Installation

### Requirements:
  - dd : 3.38.1

### Develop Build (inplace) 

Create IMAS wrapper

```{bash}
export IMAS_DD_PATH=<IMAS DD Path>
export FYTOK_MAPPING_PATH=<FYTOK_MAPPING_PATH>

cd $IMAS_DD_PATH
patch -s -p0 < <fytok dir>/setup_helper/dd_<dd version>.patch


cd <fytok dir>
python3 setup.py install_imas_wrapper --prefix=./python
```
 
### Release Build

TODO:...

```{bash}
cd <fytok dir>
python3 setup.py bist_
```

## Instructions

 
## Contribution




##  Feature
 
 
