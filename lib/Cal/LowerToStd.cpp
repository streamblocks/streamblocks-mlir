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
#include "Cal/CalOps.h"
#include "Cal/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Return `true` if all elements are of the given type.
template <typename U> bool all_of_type(ArrayRef<Value> range) {
  return all_of(range, [](Value elem) { return elem.getType().isa<U>(); });
}

//===----------------------------------------------------------------------===//
// CalTyppeConverter
//===----------------------------------------------------------------------===//

class CalTypeConverter : public TypeConverter {
public:
  CalTypeConverter() {
    addConversion([&](streamblocks::cal::IntType type) {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([&](streamblocks::cal::LogicalType type) {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([&](streamblocks::cal::RealType type) {
      return FloatType::getF32(type.getContext());
    });
  }
};

struct ConstantOpLowering
    : public OpConversionPattern<streamblocks::cal::ConstantOp> {
  using OpConversionPattern<streamblocks::cal::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(streamblocks::cal::ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CalToStdLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct CalToStdLoweringPass
    : public PassWrapper<CalToStdLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, StandardOpsDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace.

void CalToStdLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();

  ConversionTarget target(*context);
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<streamblocks::cal::ActorOp>();

  target.addLegalDialect<streamblocks::cal::CalDialect>();
}
} // namespace