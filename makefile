
PETSC_DIR=/Users/Austin/Documents/C_Progs/petsc-3.9.1/arch-darwin-c-debug
# PETSC_DIR=/Users/Austin/Documents/C_Progs/petsc-3.7.7/arch-darwin-c-opt
#PETSC_DIR=/Users/Austin/Documents/C_Progs/petsc-3.9.1/arch-darwin-c-opt

# include ${PETSC_DIR}/lib/petsc/conf/variables
# include ${PETSC_DIR}/lib/petsc/conf/rules

Includes = -I$(PETSC_DIR)/../include -I$(PETSC_DIR)/include -I/opt/X11/include -I$(PETSC_DIR)/../include/petsc/mpiuni -I$(PETSC_DIR)/include/petscconf.h

CSD_Includes = -I constants.h -I functions.h
#-fsanitize=address
Flags = -Wall -march=native -mtune=native -std=c99

Linker= -L$(PETSC_DIR)/lib -L/opt/X11/lib -lm -ldl

LFlags= -Wall -march=native -mtune=native
csd: 
	gcc -O3 -g -o constants.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/constants.c
	gcc -O3 -o csd_main.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/csd_main.c
	gcc -O3 -o initialize.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/initialize.c
	gcc -O3 -o ion_channel.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/ion_channel.c
	gcc -O3 -o array_function.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/array_functions.c
	gcc -O3 -o update_solution.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/update_solution.c
	gcc -O3 -g -o misc_print_plot.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/misc_print_plot.c
	gcc -O3 -g -o linear_update.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/linear_update.c
	gcc -O3 -g -o grid_update.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/grid_update.c

	gcc -O3 $(LFlags)   -o csd grid_update.o constants.o linear_update.o initialize.o csd_main.o ion_channel.o array_function.o update_solution.o misc_print_plot.o $(Linker) -lpetsc -lf2clapack -lf2cblas -lX11 -ldl
	#View assembly file?
	# gcc -O3 $(LFlags)   -S csd initialize.o csd_main.o ion_channel.o array_function.o update_solution.o $(Linker) -lpetsc -lf2clapack -lf2cblas -lX11 -ldl

	rm csd_main.o constants.o grid_update.o initialize.o linear_update.o ion_channel.o array_function.o update_solution.o misc_print_plot.o

debug: 
	clang -O0 -g -o csd_main.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/csd_main.c
	clang -O0 -g -o initialize.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/initialize.c
	clang -O0 -g -o ion_channel.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/ion_channel.c
	clang -O0 -g -o array_function.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/array_functions.c
	clang -O0 -g -o update_solution.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/update_solution.c
	clang -O0 -g -o misc_print_plot.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/misc_print_plot.c
	clang -O0 -g -o linear_update.o -c $(Flags) $(Includes) $(CSD_Includes) `pwd`/linear_update.c


	clang -g $(LFlags)   -o csd linear_update.o initialize.o csd_main.o ion_channel.o array_function.o update_solution.o misc_print_plot.o $(Linker) -lpetsc -lf2clapack -lf2cblas -lX11 -ldl 
	#View assembly file?
	# gcc -O3 $(LFlags)   -S csd initialize.o csd_main.o ion_channel.o array_function.o update_solution.o $(Linker) -lpetsc -lf2clapack -lf2cblas -lX11 -ldl 

	rm csd_main.o initialize.o linear_update.o ion_channel.o array_function.o update_solution.o misc_print_plot.o

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o


