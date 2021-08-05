add_mlir_dialect_library(MLIRCal
		CalTypes.cpp
        CalDialect.cpp
		CalOps.cpp
		LowerToStd.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Cal

        DEPENDS


	LINK_LIBS PUBLIC
	MLIRIR
	)
