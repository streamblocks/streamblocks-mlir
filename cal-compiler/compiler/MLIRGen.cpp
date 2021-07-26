//===- MLIRGen.cpp - MLIR Generation from a Cal AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an IR generation targeting MLIR from a
// NamespaceDecl AST for the Cal language.
//
//===----------------------------------------------------------------------===//

#include "MLIRGen.h"
#include "AST/AST.h"
#include "Cal/CalDialect.h"
#include "Cal/CalOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace streamblocks::cal;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(cal::NamespaceDecl &namespaceDecl) {

    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    StringAttr qid = StringAttr::get(builder.getContext(),
                                     namespaceDecl.getQID()->toString());
    auto theNamespace =
        builder.create<NamespaceOp>(loc(namespaceDecl.loc()), qid);

    for (auto &typeDecl : namespaceDecl.getTypeDecls()) {
      // -- TODO : Implement me
    }

    for (auto &globalVar : namespaceDecl.getvarDecls()) {
      // -- TODO : Implement me
    }

    for (auto &globalEntity : namespaceDecl.getEntityDecls()) {

      switch (globalEntity->getEntity()->getKind()) {
      case cal::Entity::Entity_Actor:
      {
        auto entity = mlirGen(cast<cal::CalActor>(*globalEntity->getEntity()));
        theNamespace.push_back(entity);
      }
        break;
      default:
        emitError(loc(globalEntity->getEntity()->loc()))
            << "MLIR codegen encountered an unhandled Global Entity kind '"
            << Twine(globalEntity->getEntity()->getKind()) << "'";
        return nullptr;
      }
    }

    theModule.push_back(theNamespace);

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, ::cal::VarDecl *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, ::cal::VarDecl *>>;

  llvm::StringMap<streamblocks::cal::ActorOp> actorMap;

  streamblocks::cal::ActorOp mlirGen(cal::CalActor &actor) {

    // Get entity name
    StringAttr name = StringAttr::get(builder.getContext(), actor.getName());

    auto inputPorts = actor.getInputs();

    auto outputPorts = actor.getOutputs();

    // Create an Actor operation
    auto theActor =
        builder.create<ActorOp>(loc(actor.loc()), name, ArrayRef<PortInfo>());

    // Is this a Process ? then create one and exit
    if (actor.getProcess() != nullptr){
      // Get actor process
      cal::ProcessDescription *process = actor.getProcess();

      // Create a Process operation
      auto theProcess = builder.create<ProcessOp>(loc(actor.getProcess()->loc()));

      // Visit its statements and create an operation for each


    }

    // State Variables

    // Actions

    // Priorities

    // Action Schedule

    return theActor;
  }

  /// Helper conversion for a Cal AST location to an MLIR location.

  mlir::Location loc(cal::location loc) {
    return mlir::FileLineColLoc::get(
        builder.getIdentifier(llvm::Twine(*loc.begin.filename)), loc.begin.line,
        loc.begin.column);
  }
};
}; // namespace

namespace streamblocks {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ::cal::NamespaceDecl &namespaceDecl) {
  return MLIRGenImpl(context).mlirGen(namespaceDecl);
}
} // namespace streamblocks