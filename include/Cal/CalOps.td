//===- CalOps.td - Cal dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CAL_OPS
#define CAL_OPS

include "CalDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cal_FooOp : CalOp<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `cal.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = cal.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

//===----------------------------------------------------------------------===//
// NamespaceOp

def NamespaceOp : CalContainer<"namespace", [
    IsolatedFromAbove,
    SymbolTable
    ]>{

    let summary = "Cal Namespace";
    let description = [{
        The "cal.namespace" operation represents an overall
        Cal Namespace, containing a list of read only variable declarations,
        functions, procedures, actors and network operations.
    }];

    let arguments = (ins);

    let verifier = "return ::verify$cppClass(*this);";

}

//===----------------------------------------------------------------------===//
// NetworkOp

def NetworkOp: CalOp<"network",
    [FunctionLike, Symbol, RegionKindInterface,
           HasParent<"mlir::ModuleOp">]>{
    let summary = "A network of actors";

    let description = [{
        The "dwf.network" operation represents an actor with a Dataflow with Firing
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
        friend class OpTrait::FunctionLike<NetworkOp>;

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


//===----------------------------------------------------------------------===//
// ActorOp

def ActorOp: CalOp<"actor", [
        HasParent<"NamespaceOp">,
        Symbol,
        FunctionLike,
        IsolatedFromAbove,
        SingleBlock,
        NoTerminator
    ]>{

    let summary = "A Cal Actor";

    let description = [{
        The "cal.actor" operation represents an actor with a Dataflow with Firing
        model of computation.
    }];

    let arguments = (ins
        ArrayAttr:$inPortNames,
        ArrayAttr:$outPortNames
    );
    let results = (outs);
    let regions = (region SizedRegion<1>: $body);

    let builders = [
        OpBuilder<(ins "StringAttr":$name)>
    ];

    let extraClassDeclaration = [{
        using FunctionLike::front;
        using FunctionLike::getBody;

        private:
            // This trait needs access to the hooks defined below.
            friend class OpTrait::FunctionLike<ActorOp>;

            /// Hooks for the input/output type enumeration in FunctionLike.
            unsigned getNumFuncArguments() { return getType().getNumInputs(); }
            unsigned getNumFuncResults() { return getType().getNumResults(); }

        public:
            /// Returns the body of a Calyx component.
            Block *getBody() { return &getOperation()->getRegion(0).front(); }

      }];

    let printer = "return ::print(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
    let verifier = "return ::verify$cppClass(*this);";
}

//===----------------------------------------------------------------------===//
// ActorOp

def ProcessOp: CalOp<"process",
    [FunctionLike, Symbol, RegionKindInterface,
           HasParent<"ActorOp">, SingleBlock, NoTerminator]>{
    let summary = "A KPN Process description";

    let description = [{
        The "dwf.process" operation represents an actor with a Dataflow with Firing
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
        static RegionKind getRegionKind(unsigned index) { return RegionKind::SSACFG;}

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
        friend class OpTrait::FunctionLike<ProcessOp>;

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



#endif // CAL_OPS
