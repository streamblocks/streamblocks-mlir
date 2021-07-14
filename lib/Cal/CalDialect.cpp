//===- CalDialect.cpp - Cal dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cal/CalDialect.h"
#include "Cal/CalOps.h"

using namespace mlir;
using namespace streamblocks::cal;

#include "Cal/CalOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Cal dialect.
//===----------------------------------------------------------------------===//

void CalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cal/CalOps.cpp.inc"
      >();
}
