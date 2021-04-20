      MODULE GLF23_SOURCE_MODULE

	private
	public :: callglf2d_TCI

	CONTAINS

	  include 'callglf2d_TCI.f'
	  include 'glf2d.f'
	  include 'c9date.f'
	  include 'r8tomsqz.f'
	  include 'smor3p.f'
	  include 'blas_zgeev.f'
	  include 'zgeev.f'

      END MODULE GLF23_SOURCE_MODULE
