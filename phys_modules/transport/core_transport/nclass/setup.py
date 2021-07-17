from numpy.distutils.core import Extension

mod_nclass = Extension(
    name='mod_nclass',
    language='f90',
    f2py_options=["--debug-capi", "only:", "call_nclass", ":"],
    sources=[
        # NOTE: Order is important!!!!
        f'src/spec_kind_mod.f90',
        f'src/nclass_mod.f90',
        f'src/call_nclass.f90',

    ],
)

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='nclass',
          version='0.0.1',
          description="Fymodule for calling the  NeoCLASSical.",
          long_description=f"""This is a `FyModule` for calling the  NeoCLASSical.    
    `FyModule` is the extended component of the tokamak integrated modeling and analysis tool `FyTok`. 
    
    For information about 'FyTok', please see 
    
        https://gitee.com/fusion_yun/fytok 
    
    or contact 
            
        Zhi YU, ASIPP, yuzhi@ipp.ac.cn

    NCLASS calculates the neoclassical transport properties of a multiple
    species axisymmetric plasma using k_order parallel and radial force
    balance equations for each species

    References:
    - S.P.Hirshman, D.J.Sigmar, Nucl Fusion 21 (1981) 1079
    - W.A.Houlberg, K.C.Shaing, S.P.Hirshman, M.C.Zarnstorff, Phys Plasmas 4 (1997) 3230
""",
          author="Zhi YU",
          author_email="yuzhi@ipp.ac.cn",
          ext_modules=[mod_nclass],
          )
