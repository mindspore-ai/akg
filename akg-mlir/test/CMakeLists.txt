configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(AKG_MLIR_TEST_DEPENDS
    akg-opt
    akg-translate
    mindspore-translate
    akg-ptx-replace
    )

add_lit_testsuite(check-akg-mlir
                  "Running the akg-mlir regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${AKG_MLIR_TEST_DEPENDS})
set_target_properties(check-akg-mlir PROPERTIES FOLDER "Tests")