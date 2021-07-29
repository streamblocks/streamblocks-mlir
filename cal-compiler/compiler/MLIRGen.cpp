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
//#include "mlir/Dialect/StandardOps/IR/Ops.h"

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
      case cal::Entity::Entity_Actor: {
        auto entity = mlirGen(cast<cal::CalActor>(*globalEntity->getEntity()));
        theNamespace.push_back(entity);
      } break;
      default:
        emitError(loc(globalEntity->getEntity()->loc()))
            << "MLIR codegen encountered an unhandled Global Entity kind '"
            << Twine(globalEntity->getEntity()->getKind()) << "'";
        return nullptr;
      }
    }

    theModule.push_back(theNamespace);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

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

  mlir::Value mlirGen(cal::ExprLiteralLong &literal) {

    // return builder.create<ConstantIntOp>(loc(literal.loc()),
    // literal.getValue(), 32);

    return builder.create<ConstantOp>(loc(literal.loc()), literal.getValue());
  }

  mlir::Value mlirGen(cal::Expression &expr) {

    switch (expr.getKind()) {
    case cal::Expression::Expr_Literal_Long:
      return mlirGen(cast<cal::ExprLiteralLong>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
    }
  }

  mlir::LogicalResult mlirGen(cal::StmtCall &call) {
    auto location = loc(call.loc());

    // Codegen the operands
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArguments()) {
      auto arg = mlirGen(*expr);
      if (!arg) {
        return mlir::failure();
      }
      operands.push_back(arg);
    }

    // Check if its a print or println
    if (operands.size() == 1) {
      if (call.getProcedure()->getKind() == cal::Expression::Expr_Variable) {
        cal::ExprVariable *callee =
            cast<cal::ExprVariable>(call.getProcedure());
        if (callee->getName() == "println") {
          builder.create<PrintlnOp>(location, operands.front());
        } else if (callee->getName() == "print") {
          builder.create<PrintOp>(location, operands.front());
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult mlirGen(cal::Statement &stmt) {
    switch (stmt.getKind()) {
    case cal::Statement::Stmt_Call:
      if (mlir::failed(mlirGen(cast<cal::StmtCall>(stmt)))) {
        return mlir::success();
      }
      break;
    default:
      emitError(loc(stmt.loc())) << "Statement not currently supported '"
                                 << Twine(stmt.getKind()) << "'";
    }

    return mlir::success();
  }

  streamblocks::cal::ActorOp mlirGen(cal::CalActor &actor) {

    // Get entity name
    StringAttr name = StringAttr::get(builder.getContext(), actor.getName());

    auto inputPorts = actor.getInputs();

    auto outputPorts = actor.getOutputs();

    // Create an Actor operation
    auto theActor =
        builder.create<ActorOp>(loc(actor.loc()), name, ArrayRef<PortInfo>());

    // Is this a Process ? then create one and exit
    if (actor.getProcess() != nullptr) {
      // Get actor process
      cal::ProcessDescription *process = actor.getProcess();

      // Create a Process operation
      BoolAttr repeat =
          BoolAttr::get(builder.getContext(), process->getRepeated());
      auto theProcess =
          builder.create<ProcessOp>(loc(actor.getProcess()->loc()), repeat);

      builder.setInsertionPointToStart(theProcess.getBody());

      // Visit its statements and create an operation for each
      for (auto &stmt : process->getStatements()) {
        if (mlir::failed(mlirGen(*stmt))) {
          return nullptr;
        }
      }

      builder.clearInsertionPoint();
      theActor.getBody()->push_back(theProcess);
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