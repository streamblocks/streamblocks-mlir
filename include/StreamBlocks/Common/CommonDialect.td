//===- CommonDialect.td - Common dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_DIALECT
#define COMMON_DIALECT

include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Common dialect definition.
//===----------------------------------------------------------------------===//

def Common_Dialect : Dialect {
    let name = "common";
    let summary = "A common dialect for dataflow dialects.";
    let description = [{
        This dialect represents the common operations and types
        shared with the different dataflow dialects.
    }];
    let cppNamespace = "::streamblocks::common";
    let extraClassDeclaration = [{
    /// Register all Cal types.
    void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base Common operation definition.
//===----------------------------------------------------------------------===//

class Common_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Common_Dialect, mnemonic, traits>{

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Common_Type : AnyTypeOf<[AnyInteger, AnyUnsignedInteger, AnyFloat, AnyComplex, FunctionType, AnyShaped, AnyVector, AnyTensor, AnyMemRef, AnyTuple]>;



#endif // COMMON_DIALECT