configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STREAMBLOCKS_TEST_DEPENDS
        FileCheck count not
        streamblocks-opt
        streamblocks-translate
        )

add_lit_testsuite(check-streamblocks "Running the streamblocks regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${STREAMBLOCKS_TEST_DEPENDS}
        )
set_target_properties(check-streamblocks PROPERTIES FOLDER "Tests")

add_lit_testsuites(STREAMBLOCKS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${STREAMBLOCKS_TEST_DEPENDS})
