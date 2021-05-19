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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "QID.h"

namespace cal {

class Expression;
class TypeExpr;
class GeneratorVarDecl;
class Decl;
class Pattern;

struct Location {
  int from_line{0};
  int from_column{0};
  int to_line{0};
  int to_column{0};
  std::string file_name;
};

class AnnotationParameter {
public:
  AnnotationParameter(Location location, std::string name,
                      std::unique_ptr<Expression> expression)
      : location(location), name(name), expression(std::move(expression)) {}
  virtual ~AnnotationParameter() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }
  Expression *getExpression() { return expression.get(); }

private:
  Location location;
  std::string name;
  std::unique_ptr<Expression> expression;
};

class Annotation {
public:
  Annotation(Location location, std::string name,
             std::vector<std::unique_ptr<AnnotationParameter>> parameters)
      : location(location), name(name), parameters(std::move(parameters)) {}

  virtual ~Annotation() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }
  llvm::ArrayRef<std::unique_ptr<AnnotationParameter>> getParameters() {
    return parameters;
  }

private:
  Location location;
  std::string name;
  std::vector<std::unique_ptr<AnnotationParameter>> parameters;
};

class Field {
public:
  Field(Location location, std::string name) : location(location), name(name) {}
  virtual ~Field() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }

private:
  Location location;
  std::string name;
};

class Generator {
public:
  Generator(Location location, std::unique_ptr<TypeExpr> type,
            std::vector<std::unique_ptr<GeneratorVarDecl>> varDecls,
            std::unique_ptr<Expression> collection)
      : location(location), type(std::move(type)),
        varDecls(std::move(varDecls)), collection(std::move(collection)) {}

  virtual ~Generator() = default;

  const Location &loc() { return location; }

  TypeExpr *getType() { return type.get(); }

  llvm::ArrayRef<std::unique_ptr<GeneratorVarDecl>> getVarDecls() {
    return varDecls;
  }

  Expression *getExpression() { return collection.get(); }

private:
  Location location;
  std::unique_ptr<TypeExpr> type;
  std::vector<std::unique_ptr<GeneratorVarDecl>> varDecls;
  std::unique_ptr<Expression> collection;
};

class Port {
public:
  Port(Location location, std::string name) : location(location), name(name) {}
  virtual ~Port() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }

private:
  Location location;
  std::string name;
};

class Variable {
public:
  Variable(Location location, std::string name)
      : location(location), name(name) {}
  virtual ~Variable() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }

private:
  Location location;
  std::string name;
};

/*
class ToolAttribute {
public:
  enum ToolAttributeKind { ToolAttribute_VALUE, ToolAttribute_TYPE };

  ToolAttribute(ToolAttributeKind kind, Location location, std::string name)
      : kind(kind), location(location), name(name) {}

  virtual ~ToolAttribute() = default;

  ToolAttributeKind getKind() const { return kind; }
  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }

private:
  const ToolAttributeKind kind;
  Location location;
  std::string name:
};

class ToolTypeAttribute : ToolAttribute {
public:
  ToolTypeAttribute(Location location, std::string name,
std::unique_ptr<TypeExpr> type) : ToolAttribute(ToolAttribute_TYPE, location,
name), type(std::move(type)) {}

  TypeExpr *getType() { return type.get(); }

  /// LLVM style RTTI
  static bool classof(const ToolAttribute *c) {
    return c->getKind() == ToolAttribute_TYPE;
  }
private:
  std::unique_ptr<TypeExpr> type;
};

class ToolValueAttribute : ToolAttribute {
public:
  ToolValueAttribute(Location location, std::string name,
std::unique_ptr<Expression> value) : ToolAttribute(ToolAttribute_TYPE, location,
name), value(std::move(value)) {}

  Expression *getValue() { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const ToolAttribute *c) {
    return c->getKind() == ToolAttribute_VALUE;
  }
private:
  std::unique_ptr<Expression> value;
};
*/

/// Base class for all type parameters nodes.
class Parameter {
public:
  enum ParameterKind { Param_Type, Param_Value };
  Parameter(ParameterKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Parameter() = default;

  ParameterKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ParameterKind kind;
  Location location;
};

class TypeParameter : Parameter {
  TypeParameter(Location loc, std::string name, std::unique_ptr<TypeExpr> value)
      : Parameter(Param_Type, loc), name(name), value(std::move(value)) {}

  std::string getName() { return name; }
  TypeExpr *getValue() { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const Parameter *c) { return c->getKind() == Param_Type; }

private:
  std::string name;
  std::unique_ptr<TypeExpr> value;
};

class ValueParameter : Parameter {
  ValueParameter(Location loc, std::string name,
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

/// Base class for all declaration nodes.
class Decl {
public:
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

  Decl(DeclKind kind, Location location, std::string name)
      : kind(kind), location(location), name(name) {}
  virtual ~Decl() = default;

  DeclKind getKind() const { return kind; }
  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }

private:
  const DeclKind kind;
  Location location;
  std::string name;
};

enum Availability {
  Availability_PUBLIC,
  Availability_PRIVATE,
  Availability_LOCAL
};

class GlobalDecl {
public:
  virtual Availability getAvailability() = 0;
};

class TypeDecl : Decl {
public:
  TypeDecl(Location location, std::string name)
      : Decl(Decl_Type, location, name) {}

  static bool classof(const Parameter *c) {
    return c->getKind() >= Decl_Type && c->getKind() <= Decl_Parameter_Type;
  }
};

class GlobalTypeDecl : TypeDecl, GlobalDecl {
public:
  GlobalTypeDecl(Location location, std::string name,
                 const Availability availability)
      : TypeDecl(location, name), availability(availability) {}

  Availability getAvailability() override { return availability; }

private:
  const Availability availability;
};

class AliasTypeDecl : GlobalTypeDecl {
public:
  AliasTypeDecl(Location location, std::string name, Availability availability,
                std::unique_ptr<TypeExpr> type)
      : GlobalTypeDecl(location, name, availability), type(std::move(type)) {}

  TypeExpr *getType() { return type.get(); }

private:
  std::unique_ptr<TypeExpr> type;
};

class ParameterTypeDecl : TypeDecl {
public:
  ParameterTypeDecl(Location location, std::string name)
      : TypeDecl(location, name) {}
};

class VarDecl : Decl {
public:
  VarDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type,
          std::unique_ptr<Expression> value, bool constant, bool external)
      : Decl(Decl_Var, location, name), type(std::move(type)),
        value(std::move(value)), constant(constant), external(external) {}

  TypeExpr *getType() const { return type.get(); }
  Expression *getValue() { return value.get(); }
  bool getConstant() const { return constant; }
  bool getExternal() const { return external; }

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
  FieldDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type,
            std::unique_ptr<Expression> value)
      : VarDecl(location, name, std::move(type), std::move(value), false,
                false) {}
};

class GeneratorVarDecl : VarDecl {
public:
  GeneratorVarDecl(Location location, std::string name)
      : VarDecl(location, name, nullptr, nullptr, true, false) {}
};

class GlobalVarDecl : VarDecl, GlobalDecl {
public:
  GlobalVarDecl(Location location, std::string name,
                std::unique_ptr<TypeExpr> type,
                std::unique_ptr<Expression> value, bool constant, bool external,
                Availability availability)
      : VarDecl(location, name, std::move(type), std::move(value), constant,
                external),
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
      : VarDecl(location, name, std::move(type), std::move(value), constant,
                false) {}
};

class ParameterVarDecl : VarDecl {
public:
  ParameterVarDecl(Location location, std::string name,
                   std::unique_ptr<TypeExpr> type,
                   std::unique_ptr<Expression> value)
      : VarDecl(location, name, std::move(type), std::move(value), false,
                false) {}
};

class PatternVarDecl : VarDecl {
  PatternVarDecl(Location location, std::string name)
      : VarDecl(location, name, nullptr, nullptr, true, false) {}
};

class PortDecl : Decl {

public:
  PortDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type)
      : Decl(Decl_Port, location, name), type(std::move(type)) {}

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
      : Decl(Decl_Variant, location, name), fields(std::move(fields)) {}

  /// LLVM style RTTI
  static bool classof(const Decl *c) { return c->getKind() == Decl_Variant; }

  llvm::ArrayRef<std::unique_ptr<FieldDecl>> getFields() { return fields; }

private:
  std::vector<std::unique_ptr<FieldDecl>> fields;
};

class AlgebraicTypeDecl {
public:
  virtual llvm::ArrayRef<std::unique_ptr<ParameterTypeDecl>>
  getTypeParameters() = 0;
  virtual llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>>
  getValueParameters() = 0;
};

class SumTypeDecl : GlobalTypeDecl, AlgebraicTypeDecl {
public:
  SumTypeDecl(Location location, std::string name, Availability availability,
              std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
              std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
              std::vector<std::unique_ptr<VariantDecl>> variants)
      : GlobalTypeDecl(location, name, availability),
        typeParameters(std::move(typeParameters)),
        valueParameters(std::move(valueParameters)),
        variants(std::move(variants)) {}

  llvm::ArrayRef<std::unique_ptr<ParameterTypeDecl>>
  getTypeParameters() override {
    return typeParameters;
  }
  llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>>
  getValueParameters() override {
    return valueParameters;
  }

  llvm::ArrayRef<std::unique_ptr<VariantDecl>> getVariants() {
    return variants;
  }

private:
  std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters;
  std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters;
  std::vector<std::unique_ptr<VariantDecl>> variants;
};

class ProductTypeDecl : GlobalTypeDecl, AlgebraicTypeDecl {
public:
  ProductTypeDecl(
      Location location, std::string name, Availability availability,
      std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
      std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
      std::vector<std::unique_ptr<FieldDecl>> fields)
      : GlobalTypeDecl(location, name, availability),
        typeParameters(std::move(typeParameters)),
        valueParameters(std::move(valueParameters)), fields(std::move(fields)) {
  }

  llvm::ArrayRef<std::unique_ptr<ParameterTypeDecl>>
  getTypeParameters() override {
    return typeParameters;
  }
  llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>>
  getValueParameters() override {
    return valueParameters;
  }

  llvm::ArrayRef<std::unique_ptr<FieldDecl>> getFields() { return fields; }

private:
  std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters;
  std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters;
  std::vector<std::unique_ptr<FieldDecl>> fields;
};

class Import {
public:
  enum Prefix { VAR, ENTITY, TYPE };

  enum ImportKind { Import_Single, Import_Group };

  Import(ImportKind kind, Location location, Prefix prefix)
      : kind(kind), location(location), prefix(prefix) {}
  virtual ~Import() = default;

  ImportKind getKind() const { return kind; }
  const Location &loc() { return location; }
  Prefix getPrefix() const { return prefix; }

private:
  const ImportKind kind;
  Location location;
  const Prefix prefix;
};

class SingleImport : Import {
public:
  SingleImport(Location location, Prefix prefix, QID globalName,
               std::string localName)
      : Import(Import_Single, location, prefix), globalName(globalName),
        localName(localName) {}

private:
  const QID globalName;
  std::string localName;
};

class GroupImport : Import {
public:
  GroupImport(Location location, Prefix prefix, QID globalName)
      : Import(Import_Group, location, prefix), globalName(globalName) {}

private:
  const QID globalName;
};

/// Base class for all type TypeExpr nodes.
class TypeExpr {
public:
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
  const TypeExprKind kind;
  Location location;
};

class NominalTypeExpr : TypeExpr {
public:
  NominalTypeExpr(Location loc, std::string name,
                  std::vector<std::unique_ptr<TypeParameter>> parameterType,
                  std::vector<std::unique_ptr<ValueParameter>> parameterValue)
      : TypeExpr(TypeExpr_Nominal, loc), name(name),
        parameterType(std::move(parameterType)),
        parameterValue(std::move(parameterValue)) {}

  llvm::StringRef getName() { return name; }
  llvm::ArrayRef<std::unique_ptr<TypeParameter>> getParameterType() {
    return parameterType;
  }
  llvm::ArrayRef<std::unique_ptr<ValueParameter>> getParameterValue() {
    return parameterValue;
  }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Nominal;
  }

private:
  std::string name;
  std::vector<std::unique_ptr<TypeParameter>> parameterType;
  std::vector<std::unique_ptr<ValueParameter>> parameterValue;
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
                    std::vector<std::unique_ptr<TypeExpr>> parameterTypes)
      : TypeExpr(TypeExpr_Procedure, loc),
        parameterTypes(std::move(parameterTypes)) {}

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getParameterTypes() {
    return parameterTypes;
  }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Procedure;
  }

private:
  std::vector<std::unique_ptr<TypeExpr>> parameterTypes;
};

class TupleTypeExpr : TypeExpr {
public:
  TupleTypeExpr(Location loc, std::vector<std::unique_ptr<TypeExpr>> types)
      : TypeExpr(TypeExpr_Tuple, loc), types(std::move(types)) {}

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
  UnionTypeExpr(TypeExprKind kind, Location location,
                std::vector<std::unique_ptr<TypeExpr>> types)
      : TypeExpr(TypeExpr_Union, location), types(std::move(types)) {}

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getTypes() { return types; }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Union;
  }

private:
  std::vector<std::unique_ptr<TypeExpr>> types;
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
    Expr_TypeConstruction,
    Expr_Case
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
  ExprLiteralDouble(Location loc, const double value)
      : Expression(Expr_Literal_Double, loc), value(value) {}
  double getValue() { return value; }

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
      : Expression(Expr_Literal_String, loc), value(value) {}
  std::string getValue() { return value; }

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
      : Expression(Expr_Literal_Bool, loc), value(value) {}
  bool getValue() { return value; }

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
      : Expression(Expr_Unary, loc), op(Op), expression(std::move(expression)) {
  }

  char getOp() { return op; }
  Expression *getExpression() { return expression.get(); }

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
  ExprApplication(Location loc, std::string callee,
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
      : Expression(Expr_Indexer, loc), structure(std::move(structure)),
        index(std::move(index)) {}

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

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_List; }

private:
  std::vector<std::unique_ptr<Expression>> elements;
};

class ExprSet : public Expression {
public:
  ExprSet(Location loc, std::vector<std::unique_ptr<Expression>> elements)
      : Expression(Expr_Set, loc), elements(std::move(elements)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Set; }

private:
  std::vector<std::unique_ptr<Expression>> elements;
};

class ExprTuple : public Expression {
public:
  ExprTuple(Location loc, std::vector<std::unique_ptr<Expression>> elements)
      : Expression(Expr_Tuple, loc), elements(std::move(elements)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }

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
      : Expression(Expr_TypeAssertion, loc), expr(std::move(expr)) {}

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
      Location loc, std::string constructor,
      std::vector<std::unique_ptr<TypeParameter>> parameterType,
      std::vector<std::unique_ptr<ValueParameter>> parameterValue,
      std::vector<std::unique_ptr<Expression>> args)
      : Expression(Expr_TypeConstruction, loc), constructor(constructor),
        parameterType(std::move(parameterType)),
        parameterValue(std::move(parameterValue)), args(std::move(args)) {}

  llvm::ArrayRef<std::unique_ptr<TypeParameter>> getParameterType() {
    return parameterType;
  }

  llvm::ArrayRef<std::unique_ptr<ValueParameter>> getParameterValue() {
    return parameterValue;
  }

  llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() { return args; }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_TypeConstruction;
  }

private:
  std::string constructor;
  std::vector<std::unique_ptr<TypeParameter>> parameterType;
  std::vector<std::unique_ptr<ValueParameter>> parameterValue;
  std::vector<std::unique_ptr<Expression>> args;
};

class ExprCase : Expression {
public:
  ExprCase(Location location, std::unique_ptr<Pattern> pattern,
           std::vector<std::unique_ptr<Expression>> guards,
           std::unique_ptr<Expression> expression)
      : Expression(Expr_Case, location), pattern(std::move(pattern)),
        guards(std::move(guards)), expression(std::move(expression)) {}

  Pattern *getPattern() { return pattern.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getGuards() { return guards; }
  Expression *getExpression() { return expression.get(); }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Case; }

private:
  std::unique_ptr<Pattern> pattern;
  std::vector<std::unique_ptr<Expression>> guards;
  std::unique_ptr<Expression> expression;
};

/// Base class for all LValue nodes.
class LValue {
public:
  enum LValueKind {
    LValue_Variable,
    LValue_Deref,
    LValue_Field,
    LValue_Indexer,
  };
  LValue(LValueKind kind, Location location) : kind(kind), location(location) {}
  virtual ~LValue() = default;

  LValueKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const LValueKind kind;
  Location location;
};

class LValueVariable : LValue {
public:
  LValueVariable(Location location, std::unique_ptr<Variable> variable)
      : LValue(LValue_Variable, location), variable(std::move(variable)) {}

  Variable *getVariable() { return variable.get(); }

  /// LLVM style RTTI
  static bool classof(const LValue *c) {
    return c->getKind() == LValue_Variable;
  }

private:
  std::unique_ptr<Variable> variable;
};

class LValueDeref : LValue {
public:
  LValueDeref(Location location, std::unique_ptr<LValueVariable> variable)
      : LValue(LValue_Deref, location), variable(std::move(variable)) {}

  LValueVariable *getVariable() { return variable.get(); }

  /// LLVM style RTTI
  static bool classof(const LValue *c) {
    return c->getKind() == LValue_Variable;
  }

private:
  std::unique_ptr<LValueVariable> variable;
};

class LValueField : LValue {
public:
  LValueField(Location location, std::unique_ptr<LValue> structure,
              std::unique_ptr<Field> field)
      : LValue(LValue_Field, location), structure(std::move(structure)),
        field(std::move(field)) {}

  LValue *getStructure() { return structure.get(); }
  Field *getField() { return field.get(); }

  /// LLVM style RTTI
  static bool classof(const LValue *c) { return c->getKind() == LValue_Field; }

private:
  std::unique_ptr<LValue> structure;
  std::unique_ptr<Field> field;
};

class LValueIndexer : LValue {
public:
  LValueIndexer(Location location, std::unique_ptr<LValue> structure,
                std::unique_ptr<Expression> index)
      : LValue(LValue_Indexer, location), structure(std::move(structure)),
        index(std::move(index)) {}

  LValue *getStructure() { return structure.get(); }
  Expression *getIndex() { return index.get(); }

  /// LLVM style RTTI
  static bool classof(const LValue *c) {
    return c->getKind() == LValue_Indexer;
  }

private:
  std::unique_ptr<LValue> structure;
  std::unique_ptr<Expression> index;
};

/// Base class for all patterns nodes.
class Pattern {
public:
  enum PatternKind {
    Pattern_Alias,
    Pattern_Alternative,
    Pattern_Binding,
    Pattern_Wildcard,
    Pattern_Deconstruction,
    Pattern_Expression,
    Pattern_List,
    Pattern_Literal,
    Pattern_Tuple,
    Pattern_Variable
  };
  Pattern(PatternKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Pattern() = default;

  PatternKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const PatternKind kind;
  Location location;
};

class PatternAlias : Pattern {
public:
  PatternAlias(Location location, std::unique_ptr<Pattern> alias,
               std::unique_ptr<Expression> expression)
      : Pattern(Pattern_Alias, location), alias(std::move(alias)),
        expression(std::move(expression)) {}

  Pattern *getPattern() { return alias.get(); }
  Expression *getExpression() { return expression.get(); }

  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Alias;
  }

private:
  std::unique_ptr<Pattern> alias;
  std::unique_ptr<Expression> expression;
};

class PatternAlternative : Pattern {
public:
  PatternAlternative(Location location,
                     std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(Pattern_Alternative, location), patterns(std::move(patterns)) {}

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() { return patterns; }

  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Alternative;
  }

private:
  std::vector<std::unique_ptr<Pattern>> patterns;
};

class PatternDeclaration {
public:
  virtual PatternVarDecl *getDeclaration() = 0;
};

class PatternBinding : Pattern, PatternDeclaration {
public:
  PatternBinding(Location location, std::unique_ptr<PatternVarDecl> declaration)
      : Pattern(Pattern_Binding, location),
        declaration(std::move(declaration)) {}

  PatternVarDecl *getDeclaration() override { return declaration.get(); }

  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Binding;
  }

private:
  std::unique_ptr<PatternVarDecl> declaration;
};

class PatternWildcard : Pattern, PatternDeclaration {
public:
  PatternWildcard(Location location) : Pattern(Pattern_Wildcard, location) {}
  PatternVarDecl *getDeclaration() override { return nullptr; }
  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Wildcard;
  }
};

class PatternDeconstruction : Pattern {
public:
  PatternDeconstruction(
      Location location, std::string deconstructor,
      std::vector<std::unique_ptr<TypeParameter>> typeParameter,
      std::vector<std::unique_ptr<ValueParameter>> valueParameter)
      : Pattern(Pattern_Deconstruction, location), deconstructor(deconstructor),
        typeParameter(std::move(typeParameter)),
        valueParameter(std::move(valueParameter)) {}

  llvm::StringRef getDeconsructor() { return deconstructor; }
  llvm::ArrayRef<std::unique_ptr<TypeParameter>> getTypeParameter() {
    return typeParameter;
  }
  llvm::ArrayRef<std::unique_ptr<ValueParameter>> getValueParameter() {
    return valueParameter;
  }

  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Deconstruction;
  }

private:
  std::string deconstructor;
  std::vector<std::unique_ptr<TypeParameter>> typeParameter;
  std::vector<std::unique_ptr<ValueParameter>> valueParameter;
};

class PatternExpression : Pattern {
public:
  PatternExpression(Location location, std::unique_ptr<Expression> expression)
      : Pattern(Pattern_Expression, location),
        expression(std::move(expression)) {}

private:
  Expression *getExpression() { return expression.get(); }
  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Expression;
  }

private:
  std::unique_ptr<Expression> expression;
};

class PatternList : Pattern {
public:
  PatternList(Location location, std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(Pattern_List, location), patterns(std::move(patterns)) {}

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() { return patterns; }

  static bool classof(const Pattern *c) { return c->getKind() == Pattern_List; }

private:
  std::vector<std::unique_ptr<Pattern>> patterns;
};

class PatternLiteral : Pattern {
public:
  PatternLiteral(Location location, std::unique_ptr<Expression> literal)
      : Pattern(Pattern_Literal, location), literal(std::move(literal)) {}

  Expression *getLiteral() { return literal.get(); }
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Literal;
  }

private:
  std::unique_ptr<Expression> literal;
};

class PatternTuple : Pattern {
public:
  PatternTuple(Location location,
               std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(Pattern_List, location), patterns(std::move(patterns)) {}

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() { return patterns; }

  static bool classof(const Pattern *c) { return c->getKind() == Pattern_List; }

private:
  std::vector<std::unique_ptr<Pattern>> patterns;
};

class PatternVariable : Pattern {
public:
  PatternVariable(Location location, std::unique_ptr<Variable> literal)
      : Pattern(Pattern_Variable, location), variable(std::move(literal)) {}

  Variable *getLiteral() { return variable.get(); }

  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Variable;
  }

private:
  std::unique_ptr<Variable> variable;
};

/// Base class for all statements nodes.
class Statement {
public:
  enum StatementKind {
    Stmt_Assignment,
    Stmt_Block,
    Stmt_Call,
    // Stmt_Case,
    Stmt_Consume,
    Stmt_Foreach,
    Stmt_If,
    Stmt_Read,
    Stmt_While,
    Stmt_Write
  };
  Statement(StatementKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Statement() = default;

  StatementKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const StatementKind kind;
  Location location;
};

class StmtAssignment : Statement {
public:
  StmtAssignment(Location location,
                 std::vector<std::unique_ptr<Annotation>> annotations,
                 std::unique_ptr<LValue> lvalue,
                 std::unique_ptr<Expression> expression)
      : Statement(Stmt_Assignment, location),
        annotations(std::move(annotations)), lvalue(std::move(lvalue)),
        expression(std::move(expression)) {}

  LValue *getLValue() { return lvalue.get(); }
  Expression *getExpression() { return expression.get(); }
  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }
  /// LLVM style RTTI
  static bool classof(const Statement *c) {
    return c->getKind() == Stmt_Assignment;
  }

private:
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::unique_ptr<LValue> lvalue;
  std::unique_ptr<Expression> expression;
};

class StmtBlock : Statement {
public:
  StmtBlock(Location location,
            std::vector<std::unique_ptr<Annotation>> annotations,
            std::vector<std::unique_ptr<TypeDecl>> typeDecls,
            std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
            std::vector<std::unique_ptr<Statement>> statements)
      : Statement(Stmt_Block, location), annotations(std::move(annotations)),
        typeDecls(std::move(typeDecls)), varDecls(std::move(varDecls)),
        statements(std::move(statements)) {}

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }

  llvm::ArrayRef<std::unique_ptr<TypeDecl>> getTypeDecls() { return typeDecls; }

  llvm::ArrayRef<std::unique_ptr<LocalVarDecl>> getVarDecls() {
    return varDecls;
  }

  llvm::ArrayRef<std::unique_ptr<Statement>> getStatements() {
    return statements;
  }
  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_Block; }

private:
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::vector<std::unique_ptr<Statement>> statements;
};

class StmtCall : Statement {
public:
  StmtCall(Location location, std::unique_ptr<Expression> procedure,
           std::vector<std::unique_ptr<Expression>> args)
      : Statement(Stmt_Call, location), procedure(std::move(procedure)),
        args(std::move(args)) {}

  Expression *getProcedure() { return procedure.get(); }

  llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() { return args; }

  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_Call; }

private:
  std::unique_ptr<Expression> procedure;
  std::vector<std::unique_ptr<Expression>> args;
};

class StmtConsume : Statement {
public:
  StmtConsume(Location location, std::unique_ptr<Port> port, int tokens)
      : Statement(Stmt_Consume, location), port(std::move(port)),
        tokens(tokens) {}

  Port *getPort() { return port.get(); }
  int getNumberOfTokens() const { return tokens; }

  /// LLVM style RTTI
  static bool classof(const Statement *c) {
    return c->getKind() == Stmt_Consume;
  }

private:
  std::unique_ptr<Port> port;
  int tokens;
};

class StmtForeach : Statement {
public:
  StmtForeach(Location location,
              std::vector<std::unique_ptr<Annotation>> annotations,
              std::unique_ptr<Generator> generator,
              std::vector<std::unique_ptr<Expression>> filters,
              std::vector<std::unique_ptr<Statement>> body)
      : Statement(Stmt_Foreach, location), annotations(std::move(annotations)),
        generator(std::move(generator)), filters(std::move(filters)),
        body(std::move(body)) {}

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }
  Generator *getGenerator() { return generator.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getFilters() { return filters; }
  llvm::ArrayRef<std::unique_ptr<Statement>> getStatement() { return body; }

  /// LLVM style RTTI
  static bool classof(const Statement *c) {
    return c->getKind() == Stmt_Foreach;
  }

private:
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::unique_ptr<Generator> generator;
  std::vector<std::unique_ptr<Expression>> filters;
  std::vector<std::unique_ptr<Statement>> body;
};

class StmtIf : Statement {
public:
  StmtIf(Location location, std::unique_ptr<Expression> condition,
         std::vector<std::unique_ptr<Statement>> thenBranch,
         std::vector<std::unique_ptr<Statement>> elseBranch)
      : Statement(Stmt_If, location), condition(std::move(condition)),
        thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}

  Expression *getCondition() { return condition.get(); }
  llvm::ArrayRef<std::unique_ptr<Statement>> getThenBranch() {
    return thenBranch;
  }
  llvm::ArrayRef<std::unique_ptr<Statement>> getElseBranch() {
    return elseBranch;
  }

  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_If; }

private:
  std::unique_ptr<Expression> condition;
  std::vector<std::unique_ptr<Statement>> thenBranch;
  std::vector<std::unique_ptr<Statement>> elseBranch;
};

class StmtRead : Statement {
public:
  StmtRead(Location location, std::unique_ptr<Port> port,
           std::vector<std::unique_ptr<LValue>> lvalues,
           std::unique_ptr<Expression> repeat)
      : Statement(Stmt_Read, location), port(std::move(port)),
        lvalues(std::move(lvalues)), repeat(std::move(repeat)) {}

  Port *getPort() { return port.get(); }
  llvm::ArrayRef<std::unique_ptr<LValue>> getLValues() { return lvalues; }
  Expression *getExpression() { return repeat.get(); }

  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_Read; }

private:
  std::unique_ptr<Port> port;
  std::vector<std::unique_ptr<LValue>> lvalues;
  std::unique_ptr<Expression> repeat;
};

class StmtWhile : Statement {
public:
  StmtWhile(Location location,
            std::vector<std::unique_ptr<Annotation>> annotations,
            std::unique_ptr<Expression> condition,
            std::vector<std::unique_ptr<Statement>> body)
      : Statement(Stmt_While, location), annotations(std::move(annotations)),
        condition(std::move(condition)), body(std::move(body)) {}

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }
  Expression *getCondition() { return condition.get(); }
  llvm::ArrayRef<std::unique_ptr<Statement>> getBody() { return body; }
  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_While; }

private:
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::unique_ptr<Expression> condition;
  std::vector<std::unique_ptr<Statement>> body;
};

class StmtWrite : Statement {
public:
  StmtWrite(Location location, std::unique_ptr<Port> port,
            std::vector<std::unique_ptr<Expression>> values,
            std::unique_ptr<Expression> repeat)
      : Statement(Stmt_Write, location), port(std::move(port)),
        values(std::move(values)), repeat(std::move(repeat)) {}

  Port *getPort() { return port.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getValues() { return values; }
  Expression *getExpression() { return repeat.get(); }

  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_Write; }

private:
  std::unique_ptr<Port> port;
  std::vector<std::unique_ptr<Expression>> values;
  std::unique_ptr<Expression> repeat;
};

/// Base class for all regexp nodes.
class RegExp {
public:
  enum RegExpKind { RegExp_Tag, RegExp_Unary, RegExp_Binary };
  RegExp(RegExpKind kind, Location location) : kind(kind), location(location) {}
  virtual ~RegExp() = default;

  RegExpKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const RegExpKind kind;
  Location location;
};

class RegExpTag : RegExp {
public:
  RegExpTag(Location location, std::unique_ptr<QID> tag)
      : RegExp(RegExp_Tag, location), tag(std::move(tag)) {}

  QID *getTag() { return tag.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Tag; }

private:
  std::unique_ptr<QID> tag;
};

class RegExpUnary : RegExp {
public:
  RegExpUnary(Location location, std::string operation,
              std::unique_ptr<RegExp> operand)
      : RegExp(RegExp_Unary, location), operation(operation),
        operand(std::move(operand)) {}

  llvm::StringRef getOperation() { return operation; }
  RegExp *getOperand() { return operand.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Unary; }

private:
  std::string operation;
  std::unique_ptr<RegExp> operand;
};

class RegExpBinary : public RegExp {
public:
  char getOp() { return op; }
  RegExp *getLHS() { return lhs.get(); }
  RegExp *getRHS() { return rhs.get(); }

  RegExpBinary(Location loc, char op, std::unique_ptr<RegExp> lhs,
               std::unique_ptr<RegExp> rhs)
      : RegExp(RegExp_Binary, loc), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Binary; }

private:
  char op;
  std::unique_ptr<RegExp> lhs, rhs;
};

/// Base class for Match node.
class Match {
public:
  Match(Location location, std::unique_ptr<InputVarDecl> declaration,
        std::unique_ptr<ExprCase> expression)
      : location(location), declaration(std::move(declaration)),
        expression(std::move(expression)) {}
  virtual ~Match() = default;

  const Location &loc() { return location; }

  InputVarDecl *getDeclaration() { return declaration.get(); }
  ExprCase *getExpression() { return expression.get(); }

private:
  Location location;
  std::unique_ptr<InputVarDecl> declaration;
  std::unique_ptr<ExprCase> expression;
};

/// Base class for Input Pattern node.
class InputPattern {
public:
  InputPattern(Location location, std::unique_ptr<Port> port,
               std::vector<std::unique_ptr<InputVarDecl>> variables,
               std::unique_ptr<Expression> repeat)
      : location(location), port(std::move(port)),
        variables(std::move(variables)), repeat(std::move(repeat)) {}

  virtual ~InputPattern() = default;
  const Location &loc() { return location; }
  llvm::ArrayRef<std::unique_ptr<InputVarDecl>> getVariables() {
    return variables;
  }
  Expression *getRepeat() { return repeat.get(); }

private:
  Location location;
  // TDDO: Replace with Match
  std::unique_ptr<Port> port;
  std::vector<std::unique_ptr<InputVarDecl>> variables;
  std::unique_ptr<Expression> repeat;
};

/// Base class for Input Pattern node.
class OutputExpression {
public:
  OutputExpression(Location location, std::unique_ptr<Port> port,
                   std::vector<std::unique_ptr<Expression>> variables,
                   std::unique_ptr<Expression> repeat)
      : location(location), port(std::move(port)), values(std::move(variables)),
        repeat(std::move(repeat)) {}

  virtual ~OutputExpression() = default;
  const Location &loc() { return location; }
  llvm::ArrayRef<std::unique_ptr<Expression>> getValues() { return values; }
  Expression *getRepeat() { return repeat.get(); }

private:
  Location location;
  // TDDO: Replace with Match
  std::unique_ptr<Port> port;
  std::vector<std::unique_ptr<Expression>> values;
  std::unique_ptr<Expression> repeat;
};

/// Base class for Action node.
class Action {
public:
  Action(Location location,
         std::vector<std::unique_ptr<Annotation>> annotations,
         std::vector<std::unique_ptr<InputPattern>> inputPatterns,
         std::vector<std::unique_ptr<OutputExpression>> outputExpressions,
         std::vector<std::unique_ptr<TypeDecl>> typeDecls,
         std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
         std::vector<std::unique_ptr<Expression>> guards,
         std::vector<std::unique_ptr<Statement>> body,
         std::unique_ptr<Expression> delay)
      : location(location), annotations(std::move(annotations)),
        inputPatterns(std::move(inputPatterns)),
        outputExpressions(std::move(outputExpressions)),
        typeDecls(std::move(typeDecls)), varDecls(std::move(varDecls)),
        guards(std::move(guards)), body(std::move(body)),
        delay(std::move(delay)) {}

  virtual ~Action() = default;
  const Location &loc() { return location; }

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }
  llvm::ArrayRef<std::unique_ptr<InputPattern>> getInputPatterns() {
    return inputPatterns;
  }
  llvm::ArrayRef<std::unique_ptr<OutputExpression>> getOutputExpressions() {
    return outputExpressions;
  }
  llvm::ArrayRef<std::unique_ptr<TypeDecl>> getTypeDecl() { return typeDecls; }
  llvm::ArrayRef<std::unique_ptr<LocalVarDecl>> getVarDecls() {
    return varDecls;
  }
  llvm::ArrayRef<std::unique_ptr<Expression>> getGuards() { return guards; }
  llvm::ArrayRef<std::unique_ptr<Statement>> getBody() { return body; }
  Expression *getDelay() { return delay.get(); }

private:
  Location location;
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::vector<std::unique_ptr<InputPattern>> inputPatterns;
  std::vector<std::unique_ptr<OutputExpression>> outputExpressions;
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::vector<std::unique_ptr<Expression>> guards;
  std::vector<std::unique_ptr<Statement>> body;
  std::unique_ptr<Expression> delay;
};

} // namespace cal

#endif