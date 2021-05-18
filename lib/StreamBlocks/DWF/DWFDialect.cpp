//===- DWFDialect.cpp - DWF dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StreamBlocks/DWF/DWFDialect.h"
#include "StreamBlocks/DWF/DWFOps.h"

using namespace mlir;
using namespace mlir::dwf;

//===----------------------------------------------------------------------===//
// DWF dialect.
//===----------------------------------------------------------------------===//

void DWFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StreamBlocks/DWF/DWFOps.cpp.inc"
      >();
}
