//===- DWFOps.td - DWF dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DWF_OPS
#define DWF_OPS

include "DWFDialect.td"


def ActorOp: DWF_Op<"actor",
    [FunctionLike, Symbol, RegionKindInterface,
           HasParent<"mlir::ModuleOp">]>{
    let summary = "An Actor";

    let description = [{
        The "dwf.actor" operation represents an actor with a Dataflow with Firing
        model of computation.
    }];

    let arguments = (ins StrArrayAttr:$argNames, StrArrayAttr:$resultNames);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);

     let skipDefaultBuilders = 1;
      let builders = [
        OpBuilder<(ins "StringAttr":$name,
                       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
      ];

    let extraClassDeclaration = [{
        using FunctionLike::front;
        using FunctionLike::getBody;

        // Implement RegionKindInterface.
        static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

        // Decode information about the input and output ports on this module.
        //SmallVector<ModulePortInfo> getPorts() {
        //  return getModulePortInfo(*this);
        //}

        // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
        // squatting on the name.
        Block *getBodyBlock() { return &body().front(); }

        // Get the module's symbolic name as StringAttr.
        StringAttr getNameAttr() {
          return (*this)->getAttrOfType<StringAttr>(
            ::mlir::SymbolTable::getSymbolAttrName());
        }

        // Get the module's symbolic name.
        StringRef getName() {
          return getNameAttr().getValue();
        }

      private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<ActorOp>;

        /// Returns the number of arguments, implementing OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getType().getInputs().size(); }
        /// Returns the number of results, implementing OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getType().getResults().size(); }

        /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
        /// attribute is present and checks if it holds a function type.  Ensures
        /// getType, getNumFuncArguments, and getNumFuncResults can be called
        ///  safely.
        LogicalResult verifyType() {
          auto type = getTypeAttr().getValue();
          if (!type.isa<FunctionType>())
            return emitOpError("requires '" + getTypeAttrName() +
                               "' attribute of function type");
          return success();
        }
      public:
      }];

    let printer = "return ::print(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
    let verifier = "return ::verify$cppClass(*this);";


}



#endif // DWF_OPS
