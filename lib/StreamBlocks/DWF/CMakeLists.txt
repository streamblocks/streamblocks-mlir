add_mlir_dialect_library(MLIRDWF
        DWFDialect.cpp
        DWFOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/StreamBlocks/DWF

        DEPENDS
        MLIRDWFOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
