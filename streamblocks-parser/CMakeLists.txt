
FIND_PACKAGE(Java)

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
        ${CMAKE_CURRENT_BINARY_DIR}/TokenMgrError.cc
        ${CMAKE_CURRENT_BINARY_DIR}/CalParserTokenManager.cc
        ${CMAKE_CURRENT_BINARY_DIR}/CharStream.cc
        ${CMAKE_CURRENT_BINARY_DIR}/ParseException.cc
        ${CMAKE_CURRENT_BINARY_DIR}/Token.cc
        ${CMAKE_CURRENT_BINARY_DIR}/CalParser.cc
        )

add_llvm_executable(streamblocks-parser ${SOURCES} ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CalParser.cc ${CMAKE_CURRENT_BINARY_DIR}/CalParser.h ${CMAKE_CURRENT_BINARY_DIR}/CalParserTokenManager.h ${CMAKE_CURRENT_BINARY_DIR}/CharStream.h ${CMAKE_CURRENT_BINARY_DIR}/JavaCC.h ${CMAKE_CURRENT_BINARY_DIR}/ParseException.h ${CMAKE_CURRENT_BINARY_DIR}/Token.h ${CMAKE_CURRENT_BINARY_DIR}/TokenMgrError.cc ${CMAKE_CURRENT_BINARY_DIR}/CalParserConstants.h ${CMAKE_CURRENT_BINARY_DIR}/CalParserTokenManager.cc ${CMAKE_CURRENT_BINARY_DIR}/CharStream.cc ${CMAKE_CURRENT_BINARY_DIR}/ErrorHandler.h ${CMAKE_CURRENT_BINARY_DIR}/ParseException.cc ${CMAKE_CURRENT_BINARY_DIR}/Token.cc ${CMAKE_CURRENT_BINARY_DIR}/TokenManager.h ${CMAKE_CURRENT_BINARY_DIR}/TokenMgrError.h
        COMMAND ${Java_JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/streamblocks-parser/external/javacc-7.0.10.jar javacc -OUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/ ${CMAKE_SOURCE_DIR}/streamblocks-parser/parser/CalParser.jj
        DEPENDS ${CMAKE_SOURCE_DIR}/streamblocks-parser/parser/CalParser.jj
        COMMENT "Generating CalParser with JavaCC"
)

add_custom_target(CalParser ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CalParser.cc)

add_dependencies(streamblocks-parser CalParser)

llvm_update_compile_flags(streamblocks-parser)
target_link_libraries(streamblocks-parser PRIVATE ${LIBS})
target_include_directories(streamblocks-parser PRIVATE ${CMAKE_CURRENT_BINARY_DIR} .)

mlir_check_all_link_libraries(streamblocks-parser)