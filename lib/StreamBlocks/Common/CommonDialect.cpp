//===- CommonDialect.cpp - Common dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StreamBlocks/Common/CommonDialect.h"
#include "StreamBlocks/Common/CommonOps.h"

using namespace mlir;
using namespace streamblocks::common;

//===----------------------------------------------------------------------===//
// DWF dialect.
//===----------------------------------------------------------------------===//

void CommonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StreamBlocks/Common/CommonOps.cpp.inc"
  >();
}
