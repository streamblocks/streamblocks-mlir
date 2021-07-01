//===- DWFDialect.td - DWF dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DWF_DIALECT
#define DWF_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// DWF dialect definition.
//===----------------------------------------------------------------------===//

def DWF_Dialect : Dialect {
    let name = "dwf";
    let summary = "A dataflow with firing actor MLIR dialect.";
    let description = [{
        This dialect represents the model of computation of a 
        dataflow with firing actor.
    }];
    let cppNamespace = "::streamblocks::dwf";
    let extraClassDeclaration = [{
    /// Register all Cal types.
    void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base DWF operation definition.
//===----------------------------------------------------------------------===//

class DWF_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<DWF_Dialect, mnemonic, traits>;

def DWF_Type : AnyTypeOf<[AnyInteger, AnyUnsignedInteger, AnyFloat, AnyComplex, FunctionType, AnyShaped, AnyVector, AnyTensor, AnyMemRef, AnyTuple]>;

#endif // DWF_DIALECT
