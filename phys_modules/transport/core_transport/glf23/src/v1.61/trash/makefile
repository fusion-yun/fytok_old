# makefile for the GLF23 test code, Jon Kinsey, GA
#   redone by DMC 12 Nov 1998 at PPPL
#   in "Lehigh University" style...
#
#   DMC 30 Mar 1999 -- removing use of -r8 or similar switches; code
#   is now explicitly REAL*8
#
#   DMC -- added NAG path for Linux NagWare.  However, this isn't working
#   as of 30 Mar 1999!
#
# for HP-UX system, the IBM RS/6000 system running AIX v4, 
#    the Solaris system, the SGI IRIX system or the DEC alpha system
#    use `gmake clean' to remove all .o files
#
# environment variable CPU is used to set platform-dependent make variables.
#

include ../../../arch.inc

ifeq ($(CPU),)
   # CPU is undefined; default to HP.
   WARNcpu = 2
   CPU = HP
endif
ifeq ($(filter $(CPU), IBM SOL SGI DEC HP NAG PGI PGI64 LF INTEL),)
   ERRcpu = 2
endif

FFLAGS = $(FFLAGS_GLF23)
FFLAGS2 = -c
FC=$(F90)

#
# Compiling and Linking commands
#
COMPILE = $(FC) $(FFLAGS) $(FFLAGS2)
LINK    = $(FC) $(FFLAGS)
#
# Targets and Rules
#

all: callglf2d.o callglf2d_DV.o zglfmex_essai25.$(EXT) zglfmex_essai50.$(EXT) zglfmex_essai100.$(EXT)

testglf xtverb::   cputst
        # without any rule, make tries to build c code
	@echo


testglf::  testglf.o callglf2d.o glf2d.o  stripx.o c9date.o r8tomsqz.o zgeev.o blas_zgeev.o
	$(LINK) -o testglf testglf.o callglf2d.o glf2d.o  stripx.o c9date.o  r8tomsqz.o zgeev.o blas_zgeev.o


stripx.o:	stripx.f
	$(COMPILE) -o stripx.o stripx.f

c9date.o:	c9date.f
	$(COMPILE) -o c9date.o c9date.f


glf2d.o:	glf2d.F glf.m f77_dcomplx.h
	$(COMPILE) -o glf2d.o glf2d.F

r8tomsqz.o:	r8tomsqz.F f77_dcomplx.h
	$(COMPILE) -o r8tomsqz.o r8tomsqz.F


callglf2d.o:	callglf2d.f glf.m
	$(COMPILE) -o callglf2d.o callglf2d.f

callglf2ddv.o:	callglf2ddv.f glf.m
	$(COMPILE) -o callglf2ddv.o callglf2ddv.f


testglf.o:	testglf.tex xtverb
	touch testglf.f
	rm testglf.f
	-./xtverb < testglf.tex > testglf.f
	$(COMPILE) -o testglf.o testglf.f


xtverb::	xtverb.f
	$(COMPILE) -o xtverb.o xtverb.f
	$(LINK) -o xtverb xtverb.o

cputst:
  ifeq ($(ERRcpu),2)
	@echo CPU=$(CPU) is not yet implemented in glf23 makefile.
	@exit 2
  endif
  ifeq ($(WARNcpu),2)
	@echo CPU defaulted, CPU=HP assumed.
  endif

zglfmex_essai25.$(EXT): zglfmex_essai25.f blas_zgeev.o callglf2d.o callglf2d_DV.o glf2d.o r8tomsqz.o zgeev.o xtverb.o
	$(MEX_F90) $^ -output zglfmex_essai25.$(EXT)

zglfmex_essai50.$(EXT): zglfmex_essai50.f blas_zgeev.o callglf2d.o callglf2d_DV.o glf2d.o r8tomsqz.o zgeev.o xtverb.o
	$(MEX_F90) $^ -output zglfmex_essai50.$(EXT)

zglfmex_essai100.$(EXT): zglfmex_essai100.f blas_zgeev.o callglf2d.o callglf2d_DV.o glf2d.o r8tomsqz.o zgeev.o xtverb.o
	$(MEX_F90) $^ -output zglfmex_essai100.$(EXT)

.PHONY: clean

clean:
	rm -rf *.o xtverb *.$(EXT)

