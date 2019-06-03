cmake_minimum_required(VERSION 3.8)
project(2d_CSD)


SET (CMAKE_C_COMPILER  "/usr/bin/clang")
set(CMAKE_CXX_STANDARD 11)

set(PETSC_DIR /Users/Austin/Documents/C_Progs/petsc-3.9.1/arch-darwin-c-debug)
#set(PETSC_DIR /Users/Austin/Documents/C_Progs/petsc-3.8.2/arch-darwin-c-opt)
#
SET( CMAKE_CXX_FLAGS  "-O3 -Wall -march=native -mtune=native" )
SET( CMAKE_EXE_LINKER_FLAGS  "-O3 -Wall -march=native -mtune=native" )
#
#SET( CMAKE_CXX_FLAGS  "-O0 -Wall -march=native -mtune=native" )
#SET( CMAKE_EXE_LINKER_FLAGS  "-O0 -Wall -march=native -mtune=native" )
#
#
#include_directories($(PETSC_DIR)/../include/
#        ${PETSC_DIR}/include/
#        /opt/X11/include/
#        ${PETSC_DIR}/../include/petsc/mpiuni/
#        ${PETSC_DIR}/include/petscconf.h
#)
#include_directories(/Users/Austin/Documents/C_Progs/petsc-3.8.2/include/
#        /Users/Austin/Documents/C_Progs/petsc-3.8.2/arch-darwin-c-debug/include/
#        /opt/X11/include/
#        /Users/Austin/Documents/C_Progs/petsc-3.8.2/include/petsc/mpiuni/
#        /Users/Austin/Documents/C_Progs/petsc-3.8.2/arch-darwin-c-debug/include/petscconf.h
#        )
#link_directories(/Users/Austin/Documents/C_Progs/petsc-3.8.2/arch-darwin-c-debug/lib/
#        /opt/X11/lib/
#)
include_directories(/Users/Austin/Documents/C_Progs/petsc-3.11.2/include/
        /Users/Austin/Documents/C_Progs/petsc-3.11.2/arch-darwin-c-opt/include/
        /opt/X11/include/
        /Users/Austin/Documents/C_Progs/petsc-3.11.2/include/petsc/mpiuni/
        /Users/Austin/Documents/C_Progs/petsc-3.11.2/arch-darwin-c-opt/include/petscconf.h
        )
link_directories(/Users/Austin/Documents/C_Progs/petsc-3.11.2/arch-darwin-c-opt/lib/
        /opt/X11/lib/
        )


set(SOURCE_FILES
        array_functions.c
        constants.h
        csd_main.c
        functions.h
        initialize.c
        ion_channel.c
        update_solution.c
        misc_print_plot.c
        linear_update.c
        constants.c
        grid_update.c)

add_executable(2d_CSD ${SOURCE_FILES})

target_link_libraries(2d_CSD petsc f2clapack f2cblas X11 dl)