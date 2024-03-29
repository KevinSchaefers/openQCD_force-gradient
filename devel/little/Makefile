################################################################################
#
# Makefile to compile and link C programs with MPI subroutines
#
# Version valid for Linux machines with MPICH
#
# "make" compiles and links the specified main programs and modules,
# using the specified libraries (if any), and produces the executables
#
# "make clean" removes all files generated by "make"
#
# Dependencies on included files are automatically taken care of
#
################################################################################

all: rmxeq mkdep mkxeq
.PHONY: all


# main programs and modules to be compiled

MAIN = check1 check2 check3 time1 time2

BLOCK = block blk_grid map_u2blk map_sw2blk map_s2blk

DIRAC = Dw_dble Dw Dw_bnd

DFL = dfl_geometry dfl_subspace

FLAGS = flags lat_parms sap_parms dfl_parms

LATTICE = bcnds blk_geometry ftidx geogen geometry uidx

LINALG = salg salg_dble valg valg_dble liealg cmatrix_dble cmatrix

LINSOLV = fgcr

LITTLE = Aw Aw_blk Aw_com1 Aw_com2 Aw_dble Aw_gen Aw_hop Aw_ops \
         ltl_modes

RANDOM = ranlux ranlux_common ranlxs ranlxd gauss

SAP = blk_solv sap_com sap sap_gcr

SFLDS = sflds scom sdcom Pbnd Pbnd_dble unorm

SU3FCTS = chexp su3prod su3ren cm3x3 random_su3

SW_TERM = pauli pauli_dble swalg swexp swflds sw_term

TCHARGE = ftcom ftensor

UFLDS = plaq_sum shift udcom uflds uinit

UTILS = endian error mutils qsum utils wspace

VFLDS = vflds vinit vcom vdcom

MODULES = $(BLOCK) $(DFL) $(DIRAC) $(FLAGS) $(LATTICE) \
          $(LINALG) $(LINSOLV) $(LITTLE) $(RANDOM) $(SAP) $(SFLDS) \
	  $(SW_TERM) $(SU3FCTS) $(TCHARGE) $(UFLDS) $(UTILS) $(VFLDS)


# Logging option (-mpilog or -mpitrace or -mpianim)

LOGOPTION =


# search path for modules

MDIR = ../../modules

VPATH = .:$(MDIR)/block:$(MDIR)/dfl:$(MDIR)/dirac:$(MDIR)/flags:\
          $(MDIR)/lattice:$(MDIR)/linalg:$(MDIR)/linsolv:$(MDIR)/little:\
          $(MDIR)/random:$(MDIR)/sap:$(MDIR)/sflds:$(MDIR)/su3fcts:\
	  $(MDIR)/sw_term:$(MDIR)/tcharge:$(MDIR)/uflds:$(MDIR)/utils:\
          $(MDIR)/vflds:


# additional include directories

INCPATH = $(MPI_INCLUDE) ../../include


# additional libraries

LIBS = m

LIBPATH = $(MPI_HOME)/lib


# scheduling and optimization options

CFLAGS = -std=c89 -pedantic -fstrict-aliasing -fopenmp -Wall \
         -Wno-long-long -Wno-format-overflow -Wstrict-prototypes -Werror \
         -O -mfpmath=sse -mno-avx -DAVX -DFMA3 -DPM

LFLAGS = -fopenmp


############################## do not change ###################################

SHELL=/bin/bash
CC=mpicc
CLINKER=$(CC)

PGMS= $(MAIN) $(MODULES)

-include $(addsuffix .d,$(PGMS))


# rule to make dependencies

$(addsuffix .d,$(PGMS)): %.d: %.c Makefile
	@ $(GCC) -ansi $< -MM $(addprefix -I,$(INCPATH)) -o $@


# rule to compile source programs

$(addsuffix .o,$(PGMS)): %.o: %.c Makefile
	$(CC) $< -c $(CFLAGS) $(LOGOPTION) $(addprefix -I,$(INCPATH))


# rule to link object files

$(MAIN): %: %.o $(addsuffix .o,$(MODULES)) Makefile
	$(CLINKER) $< $(addsuffix .o,$(MODULES)) $(LFLAGS) $(LOGOPTION) \
        $(addprefix -L,$(LIBPATH)) $(addprefix -l,$(LIBS)) -o $@


# produce executables

mkxeq: $(MAIN)


# remove old executables

rmxeq:
	@ -rm -f $(MAIN); \
        echo "delete old executables"


# make dependencies

mkdep:  $(addsuffix .d,$(PGMS))
	@ echo "generate tables of dependencies"


# clean directory

clean:
	@ -rm -rf *.d *.o *.alog *.clog *.slog $(MAIN)
.PHONY: clean

################################################################################
