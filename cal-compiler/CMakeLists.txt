
FIND_PACKAGE(FLEX)
FIND_PACKAGE(BISON)

BISON_TARGET(Parser parser/parser.yy  ${CMAKE_CURRENT_BINARY_DIR}/cal_parser.cpp)
FLEX_TARGET(Scanner parser/tokens.ll ${CMAKE_CURRENT_BINARY_DIR}/cal_lexer.yy.cpp)
ADD_FLEX_BISON_DEPENDENCY(Scanner Parser)


set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        )


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCal
        )

include_directories(include)

set(SOURCES
        parser/AST.cpp
        cal-compiler.cpp
        )

add_llvm_executable(sbc ${SOURCES} ${BISON_Parser_OUTPUTS} ${FLEX_Scanner_OUTPUTS})

llvm_update_compile_flags(sbc)
target_link_libraries(sbc
    PRIVATE
        ${LIBS}
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMToLLVMIRTranslation
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )
target_include_directories(sbc PRIVATE ${CMAKE_CURRENT_BINARY_DIR} .)

mlir_check_all_link_libraries(sbc)