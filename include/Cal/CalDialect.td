//===- CalDialect.td - Cal dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CAL_DIALECT
#define CAL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// Cal dialect definition.
//===----------------------------------------------------------------------===//

def Cal_Dialect : Dialect {
    let name = "cal";
    let summary = "A cal out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::streamblocks::cal";
    let extraClassDeclaration = [{
    /// Register all Cal types.
    void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base cal operation definition.
//===----------------------------------------------------------------------===//

class Cal_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Cal_Dialect, mnemonic, traits>;

#endif // CAL_DIALECT
