GLF23
============

References
=============

    - 2D GLF equations with massless isothermal passing electrons from
        Waltz et al, Phys. of Plasmas 6(1995)2408

Build
=============

    python -m numpy.f2py --debug-capi  -c $(pkg-config --libs openblas) src/v1.61/glf2d.F   src/v1.61/r8tomsqz.F   src/v1.61/callglf2d.f     only: callglf2d :   -m glf23_mod
    
    python -m numpy.f2py   -c $(pkg-config --libs openblas) src/glf2d.f src/c9date.f src/smor3p.f src/r8tomsqz.f src/glf.inc  only: glf2d : -m glf23_mod