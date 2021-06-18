//===- DWFOps.td - DWF dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef Common_OPS
#define Common_OPS

include "CommonDialect.td"
include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Common Operations
//===----------------------------------------------------------------------===//

def ConstantOp : Common_Op<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {

    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
            %0 = common.constant 42: i32
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "Attribute":$value),
        [{ build($_builder, $_state, value.getType(), value); }]>,
        OpBuilder<(ins "Attribute":$value, "Type":$type),
        [{ build($_builder, $_state, type, value); }]>,
    ];

    let extraClassDeclaration = [{
        Attribute getValue() { return (*this)->getAttr("value"); }

        /// Returns true if a constant operation can be built with the given value
        /// and result type.
        static bool isBuildableWith(Attribute value, Type type);
    }];

    let hasFolder = 1;
}



def AddOp : Common_Op<"add",  [NoSideEffect,
                                    SameOperandsAndResultType]> {
    let summary = "element-wise addition operation";
    let description = [{
      The "add" operation performs element-wise addition between two
      elements.
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType);

    // Specify a parser and printer method.
    let parser = [{ return ::parseBinaryOp(parser, result); }];
    let printer = [{ return ::printBinaryOp(p, *this); }];
}

#endif // Common_OPS
