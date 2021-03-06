# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Use the MFEM build directory
MFEM_DIR ?= ..
MFEM_BUILD_DIR ?= ..
SRC = $(if $(MFEM_DIR:..=),$(MFEM_DIR)/examples/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk
# Use the MFEM install directory
# SRC = $(if $(MFEM_DIR),$(MFEM_DIR)/examples/,)
# MFEM_DIR = ../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk
# TEST_MK = $(MFEM_DIR)/test.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex14 ex15 ex16 ex17
PAR_EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex8p ex9p ex10p ex11p ex12p\
 ex13p ex14p ex15p ex16p ex17p ex4D_DivSkew cfosls_parabolic cfosls_hyperbolic cfosls_wave \
 cfosls_hyperb_divfree cfosls_parab_divfree cfosls_wave_divfree cfosls_hyperbolic_anisoMG
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

MFEM_TESTS = EXAMPLES
include $(TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) 4
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example)
%-test-seq: %
	@$(call mfem-test,$<,, Serial example)

# Testing: Specific execution options
ex1-test-seq: ex1
	@$(call mfem-test,$<,, Serial example)
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example)
ex10-test-seq: ex10
	@$(call mfem-test,$<,, Serial example,-tf 5)
ex10p-test-par: ex10p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-tf 5)
ex15-test-seq: ex15
	@$(call mfem-test,$<,, Serial example,-e 1)
ex15p-test-par: ex15p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-e 1)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh displaced.mesh mesh.* ex5.mesh
	@rm -rf Example5* Example9* Example15* Example16*
	@rm -f sphere_refined.* sol.* sol_u.* sol_p.*
	@rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
	@rm -f deformed.* velocity.* elastic_energy.* mode_*
	@rm -f ex16.mesh ex16-mesh.* ex16-init.* ex16-final.*
