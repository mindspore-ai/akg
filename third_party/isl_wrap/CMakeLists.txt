set(ISL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message("ISL_DIR: ${ISL_DIR}")
message("GMP_INCLUDE_DIR: ${GMP_INCLUDE_DIR}")
message("GMP_LIBRARY: ${GMP_LIBRARY}")
message("ISL_WARP_DIR: ${ISL_WARP_DIR}")

include_directories("${ISL_WARP_DIR}/include")
include_directories("${ISL_DIR}/include")
include_directories("${ISL_DIR}")

set(ISL_SOURCES
    ${ISL_DIR}/basis_reduction_tab.c
    ${ISL_DIR}/isl_aff.c
    ${ISL_DIR}/isl_aff_map.c
    ${ISL_DIR}/isl_affine_hull.c
    ${ISL_DIR}/isl_arg.c
    ${ISL_DIR}/isl_ast_build.c
    ${ISL_DIR}/isl_ast_build_expr.c
    ${ISL_DIR}/isl_ast.c
    ${ISL_DIR}/isl_ast_codegen.c
    ${ISL_DIR}/isl_ast_graft.c
    ${ISL_DIR}/isl_bernstein.c
    ${ISL_DIR}/isl_blk.c
    ${ISL_DIR}/isl_bound.c
    ${ISL_DIR}/isl_box.c
    ${ISL_DIR}/isl_coalesce.c
    ${ISL_DIR}/isl_constraint.c
    ${ISL_DIR}/isl_convex_hull.c
    ${ISL_DIR}/isl_ctx.c
    ${ISL_DIR}/isl_deprecated.c
    ${ISL_DIR}/isl_dim_map.c
    ${ISL_DIR}/isl_equalities.c
    ${ISL_DIR}/isl_factorization.c
    ${ISL_DIR}/isl_farkas.c
    ${ISL_DIR}/isl_flow.c
    ${ISL_DIR}/isl_fold.c
    ${ISL_DIR}/isl_hash.c
    ${ISL_DIR}/isl_id.c
    ${ISL_DIR}/isl_id_to_ast_expr.c
    ${ISL_DIR}/isl_id_to_id.c
    ${ISL_DIR}/isl_id_to_pw_aff.c
    ${ISL_DIR}/isl_ilp.c
    ${ISL_DIR}/isl_input.c
    ${ISL_DIR}/isl_local.c
    ${ISL_DIR}/isl_local_space.c
    ${ISL_DIR}/isl_lp.c
    ${ISL_DIR}/isl_map.c
    ${ISL_DIR}/isl_map_list.c
    ${ISL_DIR}/isl_map_simplify.c
    ${ISL_DIR}/isl_map_subtract.c
    ${ISL_DIR}/isl_map_to_basic_set.c
    ${ISL_DIR}/isl_mat.c
    ${ISL_DIR}/isl_morph.c
    ${ISL_DIR}/isl_obj.c
    ${ISL_DIR}/isl_options.c
    ${ISL_DIR}/isl_output.c
    ${ISL_DIR}/isl_point.c
    ${ISL_DIR}/isl_polynomial.c
    ${ISL_DIR}/isl_printer.c
    ${ISL_DIR}/isl_range.c
    ${ISL_DIR}/isl_reordering.c
    ${ISL_DIR}/isl_sample.c
    ${ISL_DIR}/isl_scan.c
    ${ISL_DIR}/isl_schedule.c
    ${ISL_DIR}/isl_schedule_band.c
    ${ISL_DIR}/isl_schedule_node.c
    ${ISL_DIR}/isl_schedule_read.c
    ${ISL_DIR}/isl_schedule_tree.c
    ${ISL_DIR}/isl_scheduler.c
    ${ISL_DIR}/isl_schedule_constraints.c
    ${ISL_DIR}/isl_seq.c
    ${ISL_DIR}/isl_set_list.c
    ${ISL_DIR}/isl_set_to_ast_graft_list.c
    ${ISL_DIR}/isl_sort.c
    ${ISL_DIR}/isl_space.c
    ${ISL_DIR}/isl_stream.c
    ${ISL_DIR}/isl_stride.c
    ${ISL_DIR}/isl_tab.c
    ${ISL_DIR}/isl_tab_pip.c
    ${ISL_DIR}/isl_tarjan.c
    ${ISL_DIR}/isl_transitive_closure.c
    ${ISL_DIR}/isl_union_map.c
    ${ISL_DIR}/isl_val.c
    ${ISL_DIR}/isl_vec.c
    ${ISL_DIR}/isl_version.c
    ${ISL_DIR}/isl_vertices.c
    ${ISL_DIR}/print.c
    ${ISL_DIR}/isl_gmp.c
    ${ISL_DIR}/isl_val_gmp.c
)

# The isl library is included in the (shared)  library.
set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIC ${CMAKE_C_FLAGS}")
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -D_FORTIFY_SOURCE=2")
endif()
add_library(isl_fixed STATIC ${ISL_SOURCES})
set_property(TARGET isl_fixed PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  isl_fixed
  PRIVATE
  ${ISL_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${GMP_INCLUDE_DIR}
)
target_link_libraries(isl_fixed ${GMP_LIBRARY})

# In order to print path as relative path in log, redefine macro __FILE__ as relative path
function(file_macro_to_relative_path target_name)
    get_target_property(source_files "${target_name}" SOURCES)
    foreach(source_file ${source_files})
        get_property(definitions SOURCE "${source_file}" PROPERTY COMPILE_DEFINITIONS)
        get_filename_component(abs_path "${source_file}" ABSOLUTE)
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" relative_path ${abs_path})
        list(APPEND definitions "__FILE__=\"${relative_path}\"")
        set_property(SOURCE "${source_file}" PROPERTY COMPILE_DEFINITIONS ${definitions})
    endforeach()
endfunction()
add_definitions(-Wno-builtin-macro-redefined)
file_macro_to_relative_path(isl_fixed)

# The following sets "gitversion.h" only at configure time.
execute_process(
  COMMAND ${GIT_EXECUTABLE} describe
  WORKING_DIRECTORY ${ISL_DIR}
  OUTPUT_VARIABLE ISL_GIT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file("${ISL_WARP_DIR}/gitversion.h.in" "gitversion.h")

install(TARGETS isl_fixed)
file(GLOB HEAD_FILE_LIST ${ISL_DIR}/*.h)
install(FILES ${HEAD_FILE_LIST} DESTINATION include)
install(DIRECTORY ${ISL_DIR}/include DESTINATION include)