project(AMReX-PlotfileTools
   DESCRIPTION "AMReX tools to postprocess plot files"
   )

# Plotfile tools requires only AMReX sources from Src/Base which get always
# compiled, so no need to check which part of AMReX is enabled by user


# General Target to build all tools at once
add_custom_target(plotfile_tools)

set(_exe_names
   fboxinfo
   fcompare
   fextract
   fextrema
   fnan
   fsnapshot
   ftime
   fvarnames
   )

# Build targets one by one
foreach( _exe IN LISTS _exe_names)
   add_executable(${_exe} ${_exe}.cpp)
   target_link_libraries(${_exe} PRIVATE amrex)
   add_dependencies(plotfile_tools ${_exe})  
endforeach()


# target snapshot needs a special treatment
target_include_directories(fsnapshot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(fsnapshot PRIVATE AMReX_PPMUtil.H AMReX_PPMUtil.cpp)

# Installation
install(TARGETS ${_exe_names} RUNTIME DESTINATION bin)
