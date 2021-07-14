add_mlir_dialect_library(MLIRCal
        CalDialect.cpp
        CalOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Cal

        DEPENDS
        MLIRCalOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
