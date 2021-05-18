//===- AST.h - Node definition for the CAL AST ----------------------------===//
//
// Part of the StreamBlocks Project, under the Apache License v2.0
// with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license
// information. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the CAL language.
// The AST forms a tree structure where each node references its children
// using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef INC_CAL_AST_H_
#define INC_CAL_AST_H_

#include "Visitor.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

struct Location {
  int from_line{0};
  int from_column{0};
  int to_line{0};
  int to_column{0};
  std::string file_name;
};

namespace cal {

/// Base class for all type expression nodes.
class TypeExpr {
  enum TypeExprKind {
    TypeExpr_Nominal,
    TypeExpr_Function,
    TypeExpr_Procedure,
    TypeExpr_Tuple,
    TypeExpr_Union
  };
  TypeExpr(TypeExprKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~TypeExpr() = default;

  TypeExprKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExpressionKind kind;
  Location location;
};

class NominalTypeExpr : TypeExpr {
public:
  NominalTypeExpr(Location loc, const std::string &name,
                  std::vector<std::unique_ptr<ParameterType>> parameterType,
                  std::vector<std::unique_ptr<ParameterValue>> parameterValue)
      : TypeExpr(TypeExpr_Nominal, loc), name(name),
        parameterType(std::move(parameterType)),
        parameterValue(std::move(parameterValue)) {}

  llvm::StringRef getName() { return name; }
  llvm::ArrayRef<std::unique_ptr<ParameterType>> getParameterType() {
    return parameterType;
  }
  llvm::ArrayRef<std::unique_ptr<ParameterValue>> getParameterValue() {
    return parameterValue;
  }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Nominal;
  }

private:
  std::string name;
  std::vector<std::unique_ptr<ParameterType>> parameterType;
  std::vector<std::unique_ptr<ParameterValue>> parameterValue;
};

class FunctionTypeExpr : TypeExpr {
public:
  FunctionTypeExpr(Location loc,
                   std::vector<std::unique_ptr<TypeExpr>> parameterTypes,
                   std::unique_ptr<TypeExpr> returnType)
      : TypeExpr(TypeExpr_Function, loc),
        parameterTypes(std::move(parameterTypes)),
        returnType(std::move(returnType)) {}

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getParameterTypes() {
    return parameterTypes;
  }
  TypeExpr *getReturnType() { return returnType.get(); }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Function;
  }

private:
  std::vector<std::unique_ptr<TypeExpr>> parameterTypes;
  std::unique_ptr<TypeExpr> returnType;
};

class ProcedureTypeExpr : TypeExpr {
public:
  ProcedureTypeExpr(Location loc,
                    std::vector<std::unique_ptr<TypeExpr>> &parameterTypes)
      : TypeExpr(TypeExpr_Procedure, loc), parameterTypes(parameterTypes) {}

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getParameterTypes() {
    return parameterTypes;

    /// LLVM style RTTI
    static bool classof(const TypeExpr *c) {
      return c->getKind() == TypeExpr_Procedure;
    }

  private:
    std::vector<std::unique_ptr<TypeExpr>> parameterTypes;
  };

  class TupleTypeExpr : TypeExpr {
  public:
    TupleTypeExpr(Location loc, std::vector<std::unique_ptr<TypeExpr>> &types)
        : TypeExpr(TypeExpr_Tuple, loc), types(types) {}

    llvm::ArrayRef<std::unique_ptr<TypeExpr>> getTypes() { return types; }

    /// LLVM style RTTI
    static bool classof(const TypeExpr *c) {
      return c->getKind() == TypeExpr_Tuple;
    }

  private:
    std::vector<std::unique_ptr<TypeExpr>> types;
  };

  class UnionTypeExpr : TypeExpr {
  public:
    UnionTypeExpr(TypeExprKind kind, const Location &location,
                  const std::vector<std::unique_ptr<TypeExpr>> &types)
        : TypeExpr(TypeExpr_Union, location), types(types) {}
    llvm::ArrayRef<std::unique_ptr<TypeExpr>> getTypes() { return types; }

    /// LLVM style RTTI
    static bool classof(const TypeExpr *c) {
      return c->getKind() == TypeExpr_Union;
    }

  private:
    std::vector<std::unique_ptr<TypeExpr>> types;
  };

  /// Base class for all declaration nodes.
  class Decl {
    enum DeclKind {
      Decl_Type,
      Decl_Global_Type,
      Decl_Global_Type_Alias,
      Decl_Global_Type_Sum,
      Decl_Global_Type_Product,
      Decl_Parameter_Type,
      Decl_Var,
      Decl_Field,
      Decl_Generator_Var,
      Decl_Global_Var,
      Decl_Input_Var,
      Decl_Local_Var,
      Decl_Parameter_Var,
      Decl_Pattern_Var,
      Decl_Port,
      Decl_Variant,
      Decl_Global_Entity
    };

  public:
    Decl(DeclKind kind, Location location, std::string name)
        : kind(kind), location(location), name(name) {}

  private:
    virtual ~Decl() = default;

    DeclKind getKind() const { return kind; }
    const Location &loc() { return location; }
    llvm::StringRef getName() { return name; }

  private:
    const DeclKind kind;
    Location location;
    std::string name;
  };

  class GlobalDecl {
  public:
    enum Availability { PUBLIC, PRIVATE, LOCAL };

    virtual Availability getAvailability() = 0;
  };

  class TypeDecl : Decl {
  public:
    TypeDecl(Location location, const std::string name)
        : Decl(Decl_Type, location, name) {}

    static bool classof(const Parameter *c) {
      return c->getKind() >= Decl_Type && c->getKind() <= Decl_Parameter_Type;
    }
  };

  class GlobalTypeDecl : TypeDecl, GlobalDecl {
  public:
    GlobalTypeDecl(Location location, const std::string name,
                   const Availability availability)
        : TypeDecl(location, name), availability(availability) {}

  private:
    const Availability availability;
  };

  class AliasTypeDecl : GlobalTypeDecl {
  public:
    AliasTypeDecl(Location location, std::string name,
                  Availability availability, std::unique_ptr<TypeExpr> type)
        : GlobalTypeDecl(location, name, availability), type(type) {}

    TypeExpr *getType() { return type.get(); }

  private:
    std::unique_ptr<TypeExpr> type;
  };

  class ParameterTypeDecl : TypeDecl {
  public:
    ParameterTypeDecl(Location location, const std::string &name)
        : TypeDecl(location, name) {}
  };

  class VarDecl : Decl {
  public:
    VarDecl(Location &location, std::string name,
            std::unique_ptr<TypeExpr> type, std::unique_ptr<Expression> value,
            bool constant, bool external)
        : Decl(Decl_Var, location, name), type(type), value(value),
          constant(constant), external(external) {}

    TypeExpr *getType() const { return type.get(); }
    Expression *getValue() { return value.get(); }
    const bool getConstant() const { return constant; }
    const bool getExternal() const { return external; }

    static bool classof(const Parameter *c) {
      return c->getKind() >= Decl_Var && c->getKind() <= Decl_Pattern_Var;
    }

  private:
    std::unique_ptr<TypeExpr> type;
    std::unique_ptr<Expression> value;
    const bool constant;
    const bool external;
  };

  class FieldDecl : VarDecl {
  public:
    FieldDecl(Location location, const std::string name,
              const std::unique_ptr<TypeExpr> type,
              const std::unique_ptr<Expression> value)
        : VarDecl(location, name, type, value, false, false) {}
  };

  class GeneratorVarDecl : VarDecl {
  public:
    GeneratorVarDecl(Location location, const std::string name)
        : VarDecl(location, name, nullptr, nullptr, true, false) {}
  };

  class GlobalVarDecl : VarDecl, GlobalDecl {
  public:
    GlobalVarDecl(Location location, Availability availability,
                  std::string name, std::unique_ptr<TypeExpr> type,
                  std::unique_ptr<Expression> value)
        : VarDecl(location, name, type, value, true, false),
          availability(availability) {}

    Availability getAvailability() override { return availability; }

  private:
    const Availability availability;
  };

  class InputVarDecl : VarDecl {
  public:
    InputVarDecl(Location location, std::string name)
        : VarDecl(location, name, nullptr, nullptr, true, false) {}
  };

  class LocalVarDecl : VarDecl {
  public:
    LocalVarDecl(Location location, std::string name,
                 std::unique_ptr<TypeExpr> type,
                 std::unique_ptr<Expression> value, bool constant)
        : VarDecl(location, name, type, value, constant, falses) {}
  };

  class ParameterVarDecl : VarDecl {
  public:
    ParameterVarDecl(Location location, std::string &name,
                     std::unique_ptr<TypeExpr> type,
                     std::unique_ptr<Expression> value)
        : VarDecl(location, name, type, value, false, false) {}
  };

  class PatternVarDecl : VarDecl {
    PatternVarDecl(Location location, std::string &name)
        : VarDecl(location, name, nullptr, nullptr, true, false) {}
  };

  class PortDecl : Decl {

  public:
    PortDecl(Location location, std::string name,
             std::unique_ptr<TypeExpr> type)
        : Decl(Decl_Port, location, name), type(type) {}

  private:
    TypeExpr *getType() { return type.get(); }

    /// LLVM style RTTI
    static bool classof(const Decl *c) { return c->getKind() == Decl_Port; }

  private:
    std::unique_ptr<TypeExpr> type;
  };

  class VariantDecl : Decl {
  public:
    VariantDecl(Location location, std::string name,
                std::vector<std::unique_ptr<FieldDecl>> fields)
        : Decl(Decl_Field, location, name), fields(fields) {}

    /// LLVM style RTTI
    static bool classof(const Decl *c) { return c->getKind() == Decl_Variant; }

    llvm::ArrayRef<std::unique_ptr<FieldDecl>> getFields() { return fields; }

  private:
    std::vector<std::unique_ptr<FieldDecl>> fields;
  };

  class AlgebraicTypeDecl : GlobalTypeDecl {
  public:
    AlgebraicTypeDecl(
        const Location &location, const std::string &name,
        const Availability availability,
        std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
        std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters)
        : GlobalTypeDecl(location, name, availability),
          typeParameters(std::move(typeParameters)),
          valueParameters(std::move(valueParameters)) {}

    llvm::ArrayRef<std::unique_ptr<ParameterTypeDecl>> getTypeParameters() {
      return typeParameters;
    }
    llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>> getValueParameters() {
      return valueParameters;
    }

  private:
    std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters;
    std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters;
  };

  class SumTypeDecl : AlgebraicTypeDecl {
  public:
    SumTypeDecl(Location location, std::string name, Availability availability,
                std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
                std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
                std::vector<std::unique_ptr<VariantDecl>> variants)
        : AlgebraicTypeDecl(location, name, availability, typeParameters,
                            valueParameters),
          variants(variants) {}

    llvm::ArrayRef<std::unique_ptr<VariantDecl>> getVariants() {
      return variants;
    }

  private:
    std::vector<std::unique_ptr<VariantDecl>> variants;
  };

  class ProductTypeDecl : AlgebraicTypeDecl {
  public:
    ProductTypeDecl(
        Location location, std::string name, Availability availability,
        std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
        std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
        std::vector<std::unique_ptr<FieldDecl>> fields)
        : AlgebraicTypeDecl(location, name, availability, typeParameters,
                            valueParameters),
          fields(fields) {}

  private:
    llvm::ArrayRef<std::unique_ptr<FieldDecl>> getFields() { return fields; }

  private:
    std::vector<std::unique_ptr<FieldDecl>> fields;
  };

  /// Base class for all type parameters nodes.
  class Parameter {
    enum ParameterKind { Param_Type, Param_Value };
    TypeExpr(ParameterKind kind, Location location)
        : kind(kind), location(location) {}
    virtual ~Parameter() = default;

    ParameterKind getKind() const { return kind; }

    const Location &loc() { return location; }

  private:
    const ParameterKind kind;
    Location location;
  };

  class ParameterType : Parameter {
    ParameterType(Location loc, std::string name,
                  std::unique_ptr<TypeExpr> value)
        : Parameter(Param_Type, loc), name(name), value(std::move(value)) {}

    std::string getName() { return name; }
    TypeExpr *getValue() { return value.get(); }

    /// LLVM style RTTI
    static bool classof(const Parameter *c) {
      return c->getKind() == Param_Type;
    }

  private:
    std::string name;
    std::unique_ptr<TypeExpr> value;
  }

  class ParameterValue : Parameter {
    ParameterValue(Location loc, std::string name,
                   std::unique_ptr<Expression> value)
        : Parameter(Param_Value, loc), name(name), value(std::move(value)) {}
    llvm::StringRef getName() { return name; }
    Expression *getValue() { return value.get(); }

    /// LLVM style RTTI
    static bool classof(const Parameter *c) {
      return c->getKind() == Param_Value;
    }

  private:
    std::string name;
    std::unique_ptr<Expression> value;
  };

  /// Base class for all expressions nodes.
  class Expression {
  public:
    enum ExpressionKind {
      Expr_Literal_Integer,
      Expr_Literal_Double,
      Expr_Literal_String,
      Expr_Literal_Bool,
      Expr_Variable,
      Expr_Unary,
      Expr_Binary,
      Expr_Application,
      Expr_Indexer,
      Expr_List,
      Expr_Set,
      Expr_Tuple,
      Expr_TypeAssertion,
      Expr_TypeConstruction
    };
    Expression(ExpressionKind kind, Location location)
        : kind(kind), location(location) {}
    virtual ~Expression() = default;

    ExpressionKind getKind() const { return kind; }

    const Location &loc() { return location; }

  private:
    const ExpressionKind kind;
    Location location;
  };

  class ExprLiteralInteger : public Expression {
  public:
    ExprLiteralInteger(Location loc, const int value)
        : Expression(Expr_Literal_Integer, loc), value(value) {}
    int getValue() { return value; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Literal_Integer;
    }

  private:
    const int value;
  };

  class ExprLiteralDouble : public Expression {
  public:
    ExprLiteralDouble(Location loc, const souble value)
        : ExprLiteralDouble(Expr_Literal_Double, loc), value(value) {}
    double getValue() { return text; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Literal_Double;
    }

  private:
    const double value;
  };

  class ExprLiteralString : public Expression {
  public:
    ExprLiteralString(Location loc, std::string value)
        : ExprLiteralString(Expr_Literal_String, loc), value(value) {}
    std::string getValue() { return text; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Literal_String;
    }

  private:
    const std::string value;
  };

  class ExprLiteralBool : public Expression {
  public:
    ExprLiteralBool(Location loc, const bool value)
        : ExprLiteralBool(Expr_Literal_Bool, loc), value(value) {}
    bool getValue() { return text; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Literal_Bool;
    }

  private:
    const bool value;
  };

  class ExprVariable : public Expression {
  public:
    ExprVariable(Location loc, llvm::StringRef name)
        : Expression(Expr_Variable, loc), name(name) {}
    llvm::StringRef getName() { return name; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Variable;
    }

  private:
    std::string name;
  };

  class ExprUnary : public Expression {
  public:
    ExprUnary(Location loc, char Op, std::unique_ptr<Expression> expression)
        : Expression(Expr_Unary, loc), op(Op),
          expression(std::move(expression)) {}

    char getOp() { return op; }
    Expression *getExpr() { return expr.get(); }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Unary;
    }

  private:
    char op;
    std::unique_ptr<Expression> expression;
  };

  class ExprBinary : public Expression {
  public:
    char getOp() { return op; }
    Expression *getLHS() { return lhs.get(); }
    Expression *getRHS() { return rhs.get(); }

    ExprBinary(Location loc, char op, std::unique_ptr<Expression> lhs,
               std::unique_ptr<Expression> rhs)
        : Expression(Expr_Binary, loc), op(op), lhs(std::move(lhs)),
          rhs(std::move(rhs)) {}

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Binary;
    }

  private:
    char op;
    std::unique_ptr<Expression> lhs, rhs;
  };

  class ExprApplication : public Expression {

  public:
    ExprApplication(Location loc, const std::string &callee,
                    std::vector<std::unique_ptr<Expression>> args)
        : Expression(Expr_Application, loc), callee(callee),
          args(std::move(args)) {}

    llvm::StringRef getCallee() { return callee; }
    llvm::ArrayRef<std::unique_ptr<Expression>> getArgs() { return args; }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Application;
    }

  private:
    std::string callee;
    std::vector<std::unique_ptr<Expression>> args;
  };

  class ExprIndexer : public Expression {
  public:
    ExprIndexer(Location loc, std::unique_ptr<Expression> structure,
                std::unique_ptr<Expression> index)
        : Expression(Expr_Indexer, loc), op(Op),
          structure(std::move(structure)), index(std::move(index)) {}

    Expression *getStructure() { return structure.get(); }
    Expression *getIndex() { return index.get(); }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Indexer;
    }

  private:
    std::unique_ptr<Expression> structure, index;
  };

  class ExprList : public Expression {
  public:
    ExprList(Location loc, std::vector<std::unique_ptr<Expression>> elements)
        : Expression(Expr_List, loc), elements(std::move(elements)) {}

    llvm::ArrayRef<std::unique_ptr<Expression>> getElements() {
      return elements;
    }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_List;
    }

  private:
    std::vector<std::unique_ptr<Expression>> elements;
  };

  class ExprSet : public Expression {
  public:
    ExprSet(Location loc, std::vector<std::unique_ptr<Expression>> elements)
        : Expression(Expr_Set, loc), elements(std::move(elements)) {}

    llvm::ArrayRef<std::unique_ptr<Expression>> getElements() {
      return elements;
    }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Set;
    }

  private:
    std::vector<std::unique_ptr<Expression>> elements;
  };

  class ExprTuple : public Expression {
  public:
    ExprTuple(Location loc, std::vector<std::unique_ptr<Expression>> elements)
        : Expression(Expr_Tuple, loc), elements(std::move(elements)) {}

    llvm::ArrayRef<std::unique_ptr<Expression>> getElements() {
      return elements;
    }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_Tuple;
    }

  private:
    std::vector<std::unique_ptr<Expression>> elements;
  };

  class ExprTypeAssertion : public Expression {
  public:
    ExprTypeAssertion(Location loc, std::unique_ptr<Expression> expr,
                      std::unique_ptr<TypeExpr> type)
        : Expression(Expr_TypeAssertion, loc), op(op), expr(std::move(expr)) {}

    Expression *getExpr() { return expr.get(); }
    TypeExpr *getType() { return type.get(); }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_TypeAssertion;
    }

  private:
    std::unique_ptr<Expression> expr;
    std::unique_ptr<TypeExpr> type;
  };

  class ExprTypeConstruction : Expression {
  public:
    ExprTypeConstruction(
        Location loc, const std::string &constructor,
        std::vector<std::unique_ptr<ParameterType>> parameterType,
        std::vector<std::unique_ptr<ParameterValue>> parameterValue,
        std::vector<std::unique_ptr<Expression>> args)
        : Expression(Expr_TypeConstruction, loc), name(name),
          parameterType(std::move(parameterType)),
          parameterValue(std::move(parameterValue)), args(std::move(args)) {}

    llvm::StringRef getConstructor() { return constructor; }
    llvm::ArrayRef<std::unique_ptr<ParameterType>> getParameterType() {
      return parameterType;
    }
    llvm::ArrayRef<std::unique_ptr<ParameterValue>> getParameterValue() {
      return parameterValue;
    }
    llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() {
      return parameterValue;
    }

    /// LLVM style RTTI
    static bool classof(const Expression *c) {
      return c->getKind() == Expr_TypeConstruction;
    }

  private:
    std::string name;
    std::vector<std::unique_ptr<ParameterType>> parameterType;
    std::vector<std::unique_ptr<ParameterValue>> parameterValue;
    std::vector<std::unique_ptr<Expression>> args;
  };

} // namespace cal

#endif