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
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/fymodules	 		../python/fymodules
	
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"  					../python/fytok   
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/auxiliary 	../python/fytok/auxiliary 	
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/common  	../python/fytok/common 	
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/device  	../python/fytok/device
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/diagnosis 	../python/fytok/diagnosis
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/operation 	../python/fytok/operation
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/plasma  	../python/fytok/plasma
	@$(SPHINXAPIDOC) -feMT -o "$(APIDOCDIR)"/transport 	../python/fytok/transport

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	