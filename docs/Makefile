# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

APIDOCDIR	  = _apidoc
ROOTDIR		  = ../


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile 

apidoc:
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/phys_modules	 	../phys_modules
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"  					../python/fytok   

# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/auxiliary 	../python/fytok/modules/auxiliary 	
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/common  	../python/fytok/modules/common 	
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/device  	../python/fytok/modules/device
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/diagnosis 	../python/fytok/modules/diagnosis
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/operation 	../python/fytok/modules/operation
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/plasma  	../python/fytok/modules/plasma
# @$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/modules/transport 	../python/fytok/modules/transport

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	