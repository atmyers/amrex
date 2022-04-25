set(_sources     main.cpp)
if (AMReX_CUDA)
  set(_input_files inputs  )
else ()
  set(_input_files inputs  )
endif ()

setup_test(_sources _input_files NTASKS 1)

unset(_sources)
unset(_input_files)
