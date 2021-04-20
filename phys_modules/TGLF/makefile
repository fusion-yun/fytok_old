# Makefile for ITC Fortran 90 library for JETTO
# P. Strand, elfps@chalmers.se
# W.A. Houlberg, houlbergwa@ornl.gov
# D.Harting, d.harting@fz-juelich.de
# G.Szepesi gabor.szepesi@ifp.cnr.it


# Set the environment from the top-level Makefile of JETTO
# --------------------------------------------------------
include $(JSRCPATH)/include.mk

# Set the local environment for ITCequ
# ------------------------------------
include ../include.mk

# Object files
# ------------

#TGLFISNAN = tglf_isnan.o

TGLF_NN=tglf_nn_TM

FOBJ = tglf_isnan.o      \
       tglf_modules.o    \
       tglf_pkg.o        \
       tglf_allocate.o   \
       tglf_deallocate.o \
       tglf_startup.o    \
       tglf_hermite.o    \
       tglf_inout.o      \
       tglf_setup_geometry.o \
       tglf_LS.o         \
       tglf_eigensolver.o \
       tglf_geometry.o   \
       tglf_matrix.o     \
       tglf_max.o        \
       tglf_interface.o  \
       tglf_error.o      \
       tglf_isinf.o      \
       tglf_shutdown.o   \
       tglf_read_input.o \
       tglf_multiscale_spectrum.o \
       tglf_kygrid.o     \
       tglf_run.o        \
       tglf_TM.o         \
       tglf_init_mpi.o   \
       tglf_TM_mpi.o     \
       tglf_run_mpi.o    \
       $(TGLF_NN).o

#FMATH=-r8

tglf_run_mpi.o: tglf_run.F90
	$(FC) -o tglf_run_mpi.o $(FMATH) $(FFLAGS) -DMPI_TGLF=1 -c $<


# Avoid compiling routines with large stack usage with -mp=nonuma in PGI 
# This is a workaround for stack overflows (at least in CI) 
# If TGLF has further problems with stack overflows, it might be better to
# use heap memory instead of stack (via compiler flags, or allocation)
tglf_TM_mpi.o: tglf_TM_mpi.f90
	$(FC) -o tglf_TM_mpi.o $(FMATH) $(FFLAGS) -c $<

tglf_TM.o: tglf_TM.f90
	$(FC) -o tglf_TM.o $(FMATH) $(FFLAGS) -c $<


# set PHONY targets
# -----------------
.PHONY: all realclean clean $(MASTER_RULES) $(LIBNAME)

# make 'all' option
# -----------------
all:
	($(MAKE) -C ../../ all) || exit $$?

# Set rules passed to top-level Makefile
# --------------------------------------
$(MASTER_RULES):
	@($(MAKE) -C ../../ $@) || exit $$?

# Rule for libtransport.a
# -----------------------
$(LIBNAME): $(FOBJ)
	ar vr $(LIBNAME) $?

# make 'realclean' option
# -----------------------
realclean:
	rm -f *.o *.mod $(LIBNAME) $(LIBNAME_SMALL)

# make 'clean' option
# -------------------
clean:
	rm -f *.o *.mod

