//
// Created by Endri Bezati on 30.07.21.
//

#ifndef CAL_DIALECT_PASSES_H
#define CAL_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace streamblocks {

namespace cal {

/// Create a pass for lowering to operations in the `Std` dialects
std::unique_ptr<mlir::Pass> createLowerToStdPass();

/// Create a pass for lowering operations the remaining  operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // end namespace cal
} // end namespace streamblocks

#endif // CAL_DIALECT_PASSES_H
