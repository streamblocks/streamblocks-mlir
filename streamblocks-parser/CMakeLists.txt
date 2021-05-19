
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStreamBlocks
        )
add_llvm_executable(streamblocks-parser streamblocks-parser.cpp)

llvm_update_compile_flags(streamblocks-parser)
target_link_libraries(streamblocks-parser PRIVATE ${LIBS})

mlir_check_all_link_libraries(streamblocks-opt)