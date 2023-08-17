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
cd $IMAS_DD_PATH
patch -s -p0 < <fytok dir>/python/_build_helper/dd_<dd version>.patch

export FYTOK_MAPPING_PATH=<FYTOK_MAPPING_PATH>
cd <fytok dir>/python
python3 setup.py install_imas_wrapper --prefix=.
```
 
### Release Build

TODO:...

## Instructions

 
## Contribution




##  Feature
 
 
