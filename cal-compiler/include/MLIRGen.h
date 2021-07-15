//===- MLIRGen.h - MLIR Generation from a Cal AST -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Namespace Declaration AST for the Cal language.
//
//===----------------------------------------------------------------------===//

#ifndef CAL_MLIRGEN_H_
#define CAL_MLIRGEN_H_

#include "AST/AST.h"
#include <memory>


namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace streamblocks {
class NamespaceDecl;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ::cal::NamespaceDecl &namespaceDecl);
} // namespace streamblocks

#endif // CAL_MLIRGEN_H_