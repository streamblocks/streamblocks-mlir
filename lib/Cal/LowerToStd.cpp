//====- LowerToAffineLoops.cpp - Partial lowering from Cal to Std        --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Add information
//
//===----------------------------------------------------------------------===//

#include "Cal/CalDialect.h"
#include "Cal/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

//===----------------------------------------------------------------------===//
// CalToStdLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct CalToStdLoweringPass
    : public PassWrapper<CalToStdLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, StandardOpsDialect>();
  }
  void runOnFunction() final;
};
} // end anonymous namespace.

void CalToStdLoweringPass::runOnFunction() {
  auto function = getFunction();
}

using namespace mlir;