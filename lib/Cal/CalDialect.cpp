//===- CalDialect.cpp - Cal dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cal/CalDialect.h"
#include "Cal/CalOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace streamblocks::cal;

#include "Cal/CalDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Cal dialect.
//===----------------------------------------------------------------------===//

void CalDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "Cal/Cal.cpp.inc"
      >();
}

Operation *CalDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return llvm::TypeSwitch<Type, Operation *>(type).Default(
      [&](auto type) { return builder.create<ConstantOp>(loc, type, value); });
}