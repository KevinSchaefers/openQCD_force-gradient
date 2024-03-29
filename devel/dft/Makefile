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

MAIN = check1 check2 check3 check4 check5 check6 time1

LOCAL = dpgen

DFT = dft4d dft_com dft_shuf dft_wspace fft small_dft

LATTICE = geogen geometry bcnds

FLAGS = dft4d_parms dft_parms flags lat_parms

RANDOM = ranlux ranlux_common ranlxd ranlxs gauss

SU3FCTS = chexp cm3x3 random_su3 su3prod su3ren

UFLDS = uflds uinit

UTILS = endian error mutils qsum utils

VFLDS = vinit

MODULES = $(LOCAL) $(DFT) $(LATTICE) $(FLAGS) $(RANDOM) \
          $(SU3FCTS) $(UFLDS) $(UTILS) $(VFLDS)


# Logging option (-mpilog or -mpitrace or -mpianim)

LOGOPTION =


# search path for modules

MDIR = ../../modules

VPATH = .:$(MDIR)/dft:$(MDIR)/lattice:$(MDIR)/flags:$(MDIR)/random:\
          $(MDIR)/su3fcts:$(MDIR)/uflds:$(MDIR)/utils:$(MDIR)/vflds


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
