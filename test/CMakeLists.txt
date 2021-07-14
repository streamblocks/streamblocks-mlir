configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CAL_TEST_DEPENDS
        FileCheck count not
        cal-opt
        cal-translate
        )

add_lit_testsuite(check-cal "Running the cal regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CAL_TEST_DEPENDS}
        )
set_target_properties(check-cal PROPERTIES FOLDER "Tests")

add_lit_testsuites(CAL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CAL_TEST_DEPENDS})
