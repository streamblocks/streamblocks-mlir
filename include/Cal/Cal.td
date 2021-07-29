//===- Cal.td - Cal dialect -----------*- tablegen -*-===//
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

def CalDialect : Dialect {
    let name = "cal";
    let summary = "Cal dialect.";
    let description = [{
        Cal is a general purpose dataflow programming language.
    }];
    let cppNamespace = "::streamblocks::cal";
    let extraClassDeclaration = [{
        /// Register all Cal types.
        void registerTypes();
        /// Register all Cal attributes.
        void registerAttributes();
    }];
}

//===----------------------------------------------------------------------===//
// Cal types
//===----------------------------------------------------------------------===//

def CalIntType     : Type<CPred<"$_self.isa<cal::IntType>()">, "IntType">;
def CalUIntType    : Type<CPred<"$_self.isa<cal::UIntType>()">, "UIntType">;
def CalRealType    : Type<CPred<"$_self.isa<cal::RealType>()">, "RealType">;
def CalLogicalType : Type<CPred<"$_self.isa<cal::LogicalType>()">, "LogicalType">;

def CalBool :
  Type<CPred<"$_self.isa<::LogicalType>() && $_self.cast<::LogicalType>().getWidth() == 1">, "bool">,
  BuildableType<"LogicalType::get($_builder.getContext(), 1)">;

def CalType : AnyTypeOf<[
    CalIntType,
    CalUIntType,
    CalRealType,
    CalLogicalType
]>;

def CalNumericType : AnyTypeOf<[CalIntType, CalUIntType, CalRealType]>;
//===----------------------------------------------------------------------===//
// Base cal operation definition.
//===----------------------------------------------------------------------===//

/// Base class for the operation in this dialect
class CalOp<string mnemonic, list<OpTrait> traits = []> :
    Op<CalDialect, mnemonic, traits>;

class CalContainer<string mnemonic, list<OpTrait> traits = []> :
    CalOp<mnemonic, !listconcat(traits, [
            NoRegionArguments,
            NoTerminator,
            SingleBlock
    ])> {
    let assemblyFormat = "$body attr-dict";
    let regions = (region SizedRegion<1>: $body);

    let extraClassDeclaration = [{
    /// Returns the body of a Cal container.
    Block *getBody() { return &getOperation()->getRegion(0).front(); }
    }];
}

include "CalStructure.td"

#endif // CAL_DIALECT
