//===- ActorMachine.td - Actor Machine dialect -----------*- tablegen -*-===//
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
// Actor Machine dialect definition.
//===----------------------------------------------------------------------===//

def AMDialect : Dialect {
    let name = "am";
    let summary = "Actor Machine dialect.";
    let description = [{
        Actor Machine, is an abstract machine model for actors that codifies
        the action selection process in a state machine called its controller.
        While an action's execution is a single atomic step of an actor, it
        usually requires several microsteps of an AM to evaluate and test
        (and possibly re-evaluate and re-test) conditions until it can finally
        select and execute an action.
    }];
    let cppNamespace = "::streamblocks::am";
    let extraClassDeclaration = [{
    /// Register all AM types.
    void registerTypes();
    /// Register all AM attributes.
    void registerAttributes();
    }];
}

//===----------------------------------------------------------------------===//
// Base cal operation definition.
//===----------------------------------------------------------------------===//

/// Base class for the operation in this dialect
class AMOp<string mnemonic, list<OpTrait> traits = []> :
    Op<AMDialect, mnemonic, traits>;

class AMContainer<string mnemonic, list<OpTrait> traits = []> :
    AMOp<mnemonic, !listconcat(traits, [
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

include "ActorMachineStructure.td"

#endif // AM_DIALECT
