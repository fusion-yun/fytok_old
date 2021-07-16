from numpy.distutils.core import Extension

glf_version = "v1.61"

f2py_options = ["--debug-capi",   " only: callglf2d "]

mod_glf23 = Extension(name='mod_glf23',
                      language='f77',
                      f2py_options=f2py_options,
                      sources=[f'src/{glf_version}/glf2d.F',
                               f'src/{glf_version}/r8tomsqz.F',
                               f'src/{glf_version}/callglf2d.f',
                               #    f'src/{glf_version}/glf.m',
                               f'src/{glf_version}/f77_dcomplx.h',
                               f'src/{glf_version}/blas_zgeev.f',
                               f'src/{glf_version}/zgeev.f'
                               ],
                      extra_f77_compile_args=[
                          '-Wno-maybe-uninitialized',
                          '-Wno-unused-label',
                          '-Wno-unused-variable',
                          '-Wno-unused-dummy-argument'
                      ])
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='glf23',
          version='0.0.1',
          description="Fymodule for calling the GLF23 Transport Model.",
          long_description=f"""This is a `FyModule` for calling the GLF23 Transport Model.  
    
    `FyModule` is the extended component of the tokamak integrated modeling and analysis tool `FyTok`. 
    
    For information about 'FyTok', please see 
        
        https://gitee.com/fusion_yun/fytok 
    
    or contact 
            
            Zhi YU, ASIPP, yuzhi@ipp.ac.cn

    For information on the `GLF23` source code, please see 
        
        src/{glf_version}/README 
        
    or  contact either

            Jon Kinsey, General Atomics,  kinsey@fusion.gat.com
        or
            Ron Waltz, General Atomics, waltz@fusion.gat.com

    References:
        [1] R. E. Waltz, G. M. Staebler, W. Dorland, G. W. Hammett, M. Kotschenreuther, and J. A. Konings, Phys. Plasmas 4, 2482 (1997).
        [2] R. E. Waltz and R. L. Miller, Phys. Plasmas 6, 4265 (1999).
""",
          author="Zhi YU",
          author_email="yuzhi@ipp.ac.cn",
          ext_modules=[mod_glf23]
          )
