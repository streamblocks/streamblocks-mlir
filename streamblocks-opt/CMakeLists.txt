get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStreamBlocks
        )
add_llvm_executable(streamblocks-opt streamblocks-opt.cpp)

llvm_update_compile_flags(streamblocks-opt)
target_link_libraries(streamblocks-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(streamblocks-opt)
