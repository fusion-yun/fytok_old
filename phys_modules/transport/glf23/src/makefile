# Makefile for ITC Fortran 90 library for JETTO
# P. Strand, elfps@chalmers.se
# W.A. Houlberg, houlbergwa@ornl.gov
# D.Harting, d.harting@fz-juelich.de


# Set the environment from the top-level Makefile of JETTO
# --------------------------------------------------------
include $(JSRCPATH)/include.mk

# Set the local environment for ITCequ
# ------------------------------------
include ../include.mk

# Object files
# ------------
FOBJ = GLF23_source_module.o

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
	rm -f *.o *.mod $(LIBNAME)

# make 'clean' option
# -------------------
clean:
	rm -f *.o

# Dependencies
# ------------
GLF23_source_module.o  : GLF23_source_module.f callglf2d_TCI.f glf2d.f c9date.f blas_zgeev.f zgeev.f smor3p.f r8tomsqz.f glf.m 
 
