# NCLASS

NCLASS calculates the neoclassical transport properties of a multiple
species axisymmetric plasma using k_order parallel and radial force
balance equations for each species

References:
=============

- S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
- W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997)

Build
=============

    python -m numpy.f2py --debug-capi  -c $(pkg-config --libs openblas) src/spec_kind_mod.f90 src/nclass_mod.f90 only: nclass : -m nclass_mod