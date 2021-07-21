//===- CalOps.td - Cal dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


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

def NetworkOp: CalOp<"network", [
        HasParent<"NamespaceOp">,
        Symbol,
        FunctionLike,
        IsolatedFromAbove,
        SingleBlock,
        NoTerminator
    ]>{

    let summary = "A Cal Network";

    let description = [{
        The "cal.network" operation represents a network of actors.
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
            friend class OpTrait::FunctionLike<NetworkOp>;

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
        OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports)>
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
// ProcessOp

def ProcessOp: CalContainer<"process", [
    HasParent<"ActorOp">
    ]>{
    let summary = "A KPN Process description";

    let description = [{
        The "cal.process" operation represents an actor with a Dataflow with Firing
        model of computation.
    }];

    let arguments = (ins);
    let results = (outs);
    let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// PrintOp

def PrintOp : CalOp<"print"> {
  let summary = "print operation";
  let description = [{
    The "cal.print" builtin operation prints a value, and produces
    no results.
  }];

  let arguments = (ins Variadic<AnyType>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// PrintOp

def PrintlnOp : CalOp<"println"> {
  let summary = "println operation";
  let description = [{
    The "cal.println" builtin operation prints a value and adds a new
    line to output, and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
