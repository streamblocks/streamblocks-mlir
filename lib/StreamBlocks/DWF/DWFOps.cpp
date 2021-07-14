//===- DWFOps.cpp - DWF dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StreamBlocks/DWF/DWFOps.h"
#include "StreamBlocks/DWF/DWFDialect.h"

using namespace mlir;

using namespace streamblocks::dwf;

//===----------------------------------------------------------------------===//
// NetworkOp

static ParseResult parseNetworkOp(OpAsmParser &parser, OperationState &result){
  return failure();
}

static void print(OpAsmPrinter &p, NetworkOp op) {
}

static LogicalResult verifyNetworkOp(NetworkOp op) {
  // -- TODO : Implement
  return success();
}

//===----------------------------------------------------------------------===//
// ActorOp

static ParseResult parseActorOp(OpAsmParser &parser, OperationState &result){
  return failure();
}

static void print(OpAsmPrinter &p, ActorOp op) {
}

static LogicalResult verifyActorOp(ActorOp op) {
  // -- TODO : Implement
  return success();
}

//===----------------------------------------------------------------------===//
// ProcessOp

static ParseResult parseProcessOp(OpAsmParser &parser, OperationState &result){
  return failure();
}

static void print(OpAsmPrinter &p, ProcessOp op) {
}

static LogicalResult verifyProcessOp(ProcessOp op) {
  // -- TODO : Implement
  return success();
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "StreamBlocks/DWF/DWFOps.cpp.inc"