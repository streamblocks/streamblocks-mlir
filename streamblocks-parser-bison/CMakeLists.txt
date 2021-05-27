
FIND_PACKAGE(FLEX)
FIND_PACKAGE(BISON)

BISON_TARGET(Parser ${CMAKE_SOURCE_DIR}/lib/StreamBlocks/parser/parser.yy  ${CMAKE_CURRENT_BINARY_DIR}/cal_parser.cpp)
FLEX_TARGET(Scanner ${CMAKE_SOURCE_DIR}/lib/StreamBlocks/parser/tokens.ll ${CMAKE_CURRENT_BINARY_DIR}/cal_lexer.yy.cpp)
ADD_FLEX_BISON_DEPENDENCY(Scanner Parser)


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStreamBlocks
        )

set(SOURCES
        streamblocks-parser.cpp
        )

add_llvm_executable(streamblocks-parser-bison ${SOURCES} ${BISON_Parser_OUTPUTS} ${FLEX_Scanner_OUTPUTS})

llvm_update_compile_flags(streamblocks-parser-bison)
target_link_libraries(streamblocks-parser-bison PRIVATE ${LIBS})
target_include_directories(streamblocks-parser-bison PRIVATE ${CMAKE_CURRENT_BINARY_DIR} .)

mlir_check_all_link_libraries(streamblocks-parser-bison)