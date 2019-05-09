set(CONFIG_NAME "toss_3_x86_64_ib-gcc@8.1.0" CACHE PATH "") 

set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-8.1.0/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-8.1.0/bin/g++" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.1.0/bin/gfortran" CACHE PATH "")

set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
set(ENABLE_MPI ON CACHE BOOL "" FORCE)

set(MPI_HOME             "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.1.0" CACHE PATH "")
set(MPI_C_COMPILER       "${MPI_HOME}/bin/mpicc"   CACHE PATH "")
set(MPI_CXX_COMPILER     "${MPI_HOME}/bin/mpicxx"  CACHE PATH "")
set(MPI_Fortran_COMPILER "${MPI_HOME}/bin/mpifort" CACHE PATH "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

set(SPHINX_EXECUTABLE "/usr/bin/sphinx-build" CACHE PATH "" FORCE)

set( ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "" FORCE )

set(CUDA_ENABLED      "OFF"       CACHE PATH "" FORCE)
set(CHAI_BUILD_TYPE   "cpu-no-rm" CACHE PATH "" FORCE)
set(CHAI_ARGS         ""          CACHE PATH "" FORCE)

set(ENABLE_OPENMP     "ON"        CACHE PATH "" FORCE)

set(GEOSX_TPL_ROOT_DIR "/usr/gapps/GEOS/geosx/thirdPartyLibs/" CACHE PATH "")
set(GEOSX_TPL_DIR "${GEOSX_TPL_ROOT_DIR}/install-${CONFIG_NAME}-release" CACHE PATH "")

set(SPHINX_EXECUTABLE "/usr/bin/sphinx-build" CACHE PATH "" FORCE)
set(UNCRUSTIFY_EXECUTABLE "${GEOSX_TPL_DIR}/uncrustify/bin/uncrustify" CACHE PATH "" FORCE )