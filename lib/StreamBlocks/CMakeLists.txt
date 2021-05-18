add_mlir_dialect_library(MLIRStreamBlocks
        DWF/DWFDialect.cpp
        DWF/DWFOps.cpp

        ADDITIONAL_HEADER_DIRS
		${PROJECT_SOURCE_DIR}/include/StreamBlocks/DWF
        ${PROJECT_SOURCE_DIR}/include/StreamBlocks/AST

        DEPENDS
        MLIRDWFOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
