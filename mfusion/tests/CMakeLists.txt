# We need to set MFUSION_TOOLS_DIR so lit.site.cfg.py.in can pick it up.
# LLVM_RUNTIME_OUTPUT_INTDIR is set to ${CMAKE_BINARY_DIR}/bin in top-level CMakeLists.txt
set(MFUSION_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MFUSION_TEST_DEPENDS
    mfusion-opt
    FileCheck
    count
    not
    )

add_lit_testsuite(check-mfusion
                  "Running the mfusion regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/ut
                  DEPENDS
                  ${MFUSION_TEST_DEPENDS})
set_target_properties(check-mfusion PROPERTIES FOLDER "Tests")
