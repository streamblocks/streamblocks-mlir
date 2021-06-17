add_mlir_dialect_library(MLIRCommon
        CommonDialect.cpp
        CommonOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/StreamBlocks/Common

        DEPENDS
        MLIRCommonOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
