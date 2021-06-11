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

#include "location.hh"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include "QID.h"

using Location = cal::location;

namespace cal {

class Entity;

class Expression {
public:
  enum ExpressionKind {
    Expr_Literal_Long,
    Expr_Literal_Real,
    Expr_Literal_Char,
    Expr_Literal_String,
    Expr_Literal_Bool,
    Expr_Literal_Null,
    Expr_Variable,
    Expr_Unary,
    Expr_Binary,
    Expr_Comprehension,
    Expr_Application,
    Expr_If,
    Expr_Indexer,
    Expr_Lambda,
    Expr_Let,
    Expr_List,
    Expr_Set,
    Expr_Map,
    Expr_Proc,
    Expr_Tuple,
    Expr_TypeAssertion,
    Expr_TypeConstruction,
    Expr_Case
  };
  Expression(ExpressionKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Expression() = default;

  virtual std::unique_ptr<Expression> clone() const = 0;

  ExpressionKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const ExpressionKind kind;
  Location location;
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

  virtual std::unique_ptr<Statement> clone() const = 0;

  StatementKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const StatementKind kind;
  Location location;
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

  virtual std::unique_ptr<Decl> clone() const = 0;

  DeclKind getKind() const { return kind; }
  const Location &loc() const { return location; }
  llvm::StringRef getName() const { return name; }

private:
  const DeclKind kind;
  Location location;
  std::string name;
};

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

  virtual std::unique_ptr<TypeExpr> clone() const = 0;

  TypeExprKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const TypeExprKind kind;
  Location location;
};

class Parameter {
public:
  enum ParameterKind { Param_Type, Param_Value };
  Parameter(ParameterKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Parameter() = default;

  virtual std::unique_ptr<Parameter> clone() const = 0;

  ParameterKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const ParameterKind kind;
  Location location;
};

class VarDecl : public Decl {
public:
  VarDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type,
          std::unique_ptr<Expression> value, bool constant, bool external)
      : Decl(Decl_Var, location, name), type(std::move(type)),
        value(std::move(value)), constant(constant), external(external) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<VarDecl>(
        loc(), llvm::Twine(getName()).str(),
        type != nullptr ? type->clone() : std::unique_ptr<TypeExpr>(),
        value != nullptr ? value->clone() : std::unique_ptr<Expression>(),
        constant, external);
  }

  TypeExpr *getType() { return type.get(); }
  Expression *getValue() { return value.get(); }
  bool getConstant() { return constant; }
  bool getExternal() { return external; }

  void setConstant(bool value) { constant = value; }

  void setExternal(bool value) { external = value; }

  static bool classof(const Parameter *c) {
    return c->getKind() >= Decl_Var && c->getKind() <= Decl_Pattern_Var;
  }

protected:
  std::unique_ptr<TypeExpr> type;
  std::unique_ptr<Expression> value;
  bool constant;
  bool external;
};

class GeneratorVarDecl : public VarDecl {
public:
  GeneratorVarDecl(Location location, std::string name)
      : VarDecl(location, name, nullptr, nullptr, true, false) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<GeneratorVarDecl>(loc(),
                                              llvm::Twine(getName()).str());
  }
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
            std::unique_ptr<Expression> collection,
            std::vector<std::unique_ptr<Expression>> filters)
      : location(location), type(std::move(type)),
        varDecls(std::move(varDecls)), collection(std::move(collection)),
        filters(std::move(filters)) {}

  virtual ~Generator() = default;

  std::unique_ptr<Generator> clone() const {

    std::vector<std::unique_ptr<GeneratorVarDecl>> tovVarDecls;
    tovVarDecls.reserve(varDecls.size());
    for (const auto &e : varDecls) {
      auto t = std::unique_ptr<GeneratorVarDecl>(
          static_cast<GeneratorVarDecl *>(e->clone().release()));
      tovVarDecls.push_back(std::move(t));
    }

    std::vector<std::unique_ptr<Expression>> toFilters;
    toFilters.reserve(filters.size());
    for (const auto &e : filters) {
      auto t = std::unique_ptr<Expression>(
          static_cast<Expression *>(e->clone().release()));
      toFilters.push_back(std::move(t));
    }

    return std::make_unique<Generator>(
        loc(), type->clone(), std::move(tovVarDecls), collection->clone(),
        std::move(toFilters));
  }

  void addFilter(std::unique_ptr<Expression> filter) {
    filters.push_back(std::move(filter));
  }

  const Location &loc() const { return location; }

  TypeExpr *getType() { return type.get(); }

  llvm::ArrayRef<std::unique_ptr<GeneratorVarDecl>> getVarDecls() {
    return varDecls;
  }

  Expression *getExpression() { return collection.get(); }

  llvm::ArrayRef<std::unique_ptr<Expression>> getFilters() { return filters; }

private:
  Location location;
  std::unique_ptr<TypeExpr> type;
  std::vector<std::unique_ptr<GeneratorVarDecl>> varDecls;
  std::unique_ptr<Expression> collection;
  std::vector<std::unique_ptr<Expression>> filters;
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

/// Base class for all type parameters nodes.

class TypeParameter : public Parameter {
public:
  TypeParameter(Location loc, std::string name, std::unique_ptr<TypeExpr> value)
      : Parameter(Param_Type, loc), name(name), value(std::move(value)) {}

  std::string getName() { return name; }
  TypeExpr *getValue() { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const Parameter *c) { return c->getKind() == Param_Type; }

  std::unique_ptr<Parameter> clone() const override {
    return std::make_unique<TypeParameter>(loc(), name, value->clone());
  }

private:
  std::string name;
  std::unique_ptr<TypeExpr> value;
};

class ValueParameter : public Parameter {
public:
  ValueParameter(Location loc, std::string name,
                 std::unique_ptr<Expression> value)
      : Parameter(Param_Value, loc), name(name), value(std::move(value)) {}

  std::unique_ptr<Parameter> clone() const override {
    return std::make_unique<ValueParameter>(loc(), name, value->clone());
  }

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

/// classes for all declaration nodes.

enum Availability { PUBLIC, PRIVATE, LOCAL };

class GlobalDecl {
public:
  virtual ~GlobalDecl() = default;
  virtual Availability getAvailability() const = 0;
};

class TypeDecl : public Decl {
public:
  TypeDecl(Location location, std::string name)
      : Decl(Decl_Type, location, name) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<TypeDecl>(loc(), llvm::Twine(getName()).str());
  }

  static bool classof(const Parameter *c) {
    return c->getKind() >= Decl_Type && c->getKind() <= Decl_Parameter_Type;
  }
};

class GlobalTypeDecl : public TypeDecl, GlobalDecl {
public:
  GlobalTypeDecl(Location location, std::string name,
                 const Availability availability)
      : TypeDecl(location, name), availability(availability) {}

  Availability getAvailability() const override { return availability; }

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<GlobalTypeDecl>(loc(), llvm::Twine(getName()).str(),
                                            getAvailability());
  }

private:
  const Availability availability;
};

class AliasTypeDecl : public GlobalTypeDecl {
public:
  AliasTypeDecl(Location location, std::string name, Availability availability,
                std::unique_ptr<TypeExpr> type)
      : GlobalTypeDecl(location, name, availability), type(std::move(type)) {}

  TypeExpr *getType() { return type.get(); }

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<AliasTypeDecl>(loc(), llvm::Twine(getName()).str(),
                                           getAvailability(), type->clone());
  }

private:
  std::unique_ptr<TypeExpr> type;
};

class GlobalEntityDecl : public Decl, GlobalDecl {
public:
  GlobalEntityDecl(Location location, std::string name,
                   std::unique_ptr<Entity> entity, Availability availability,
                   bool external)
      : Decl(Decl_Global_Entity, location, name), entity(std::move(entity)),
        availability(availability), external(external) {}

  std::unique_ptr<Decl> clone() const override {
    // TODO : implement
    return std::unique_ptr<Decl>();
  }

  Entity *getEntity() { return entity.get(); }
  Availability getAvailability() const override { return availability; }
  bool getExternal() const { return external; }

  static bool classof(const Decl *c) {
    return c->getKind() == Decl_Global_Entity;
  }

private:
  std::unique_ptr<Entity> entity;
  const Availability availability;
  const bool external;
};

class ParameterTypeDecl : public TypeDecl {
public:
  ParameterTypeDecl(Location location, std::string name)
      : TypeDecl(location, name) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<ParameterTypeDecl>(loc(),
                                               llvm::Twine(getName()).str());
  }
};

class FieldDecl : public VarDecl {
public:
  FieldDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type,
            std::unique_ptr<Expression> value)
      : VarDecl(location, name, std::move(type), std::move(value), false,
                false) {}
  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<FieldDecl>(loc(), llvm::Twine(getName()).str(),
                                       type->clone(), value->clone());
  }
};

class GlobalVarDecl : public VarDecl, public GlobalDecl {
public:
  GlobalVarDecl(Location location, std::string name,
                std::unique_ptr<TypeExpr> type,
                std::unique_ptr<Expression> value, bool constant, bool external,
                Availability availability)
      : VarDecl(location, name, std::move(type), std::move(value), constant,
                external),
        availability(availability) {}

  GlobalVarDecl(std::unique_ptr<VarDecl> decl, bool external,
                Availability availability)
      : VarDecl(decl->loc(), llvm::Twine(decl->getName()).str(),
                decl->getType() != nullptr ? decl->getType()->clone()
                                           : std::unique_ptr<TypeExpr>(),
                decl->getValue() != nullptr ? decl->getValue()->clone()
                                            : std::unique_ptr<Expression>(),
                decl->getConstant(), external),
        availability(availability) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<GlobalVarDecl>(loc(), llvm::Twine(getName()).str(),
                                           type->clone(), value->clone(),
                                           constant, external, availability);
  }

  Availability getAvailability() const override { return availability; }

private:
  const Availability availability;
};

class InputVarDecl : public VarDecl {
public:
  InputVarDecl(Location location, std::string name)
      : VarDecl(location, name, nullptr, nullptr, true, false) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<InputVarDecl>(loc(), llvm::Twine(getName()).str());
  }
};

class LocalVarDecl : public VarDecl {
public:
  LocalVarDecl(Location location, std::string name,
               std::unique_ptr<TypeExpr> type,
               std::unique_ptr<Expression> value, bool constant)
      : VarDecl(location, name, std::move(type), std::move(value), constant,
                false) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<LocalVarDecl>(loc(), llvm::Twine(getName()).str(),
                                          type->clone(), value->clone(),
                                          constant);
  }
};

class ParameterVarDecl : public VarDecl {
public:
  ParameterVarDecl(Location location, std::string name,
                   std::unique_ptr<TypeExpr> type,
                   std::unique_ptr<Expression> value)
      : VarDecl(location, name, std::move(type), std::move(value), false,
                false) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<ParameterVarDecl>(
        loc(), llvm::Twine(getName()).str(),
        type != nullptr ? type->clone() : std::unique_ptr<TypeExpr>(),
        value != nullptr ? value->clone() : std::unique_ptr<Expression>());
  }
};

class PatternVarDecl : public VarDecl {
public:
  PatternVarDecl(Location location, std::string name)
      : VarDecl(location, name, nullptr, nullptr, true, false) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<PatternVarDecl>(loc(),
                                            llvm::Twine(getName()).str());
  }
};

class PortDecl : public Decl {
public:
  PortDecl(Location location, std::string name, std::unique_ptr<TypeExpr> type)
      : Decl(Decl_Port, location, name), type(std::move(type)) {}

  std::unique_ptr<Decl> clone() const override {
    return std::make_unique<PortDecl>(loc(), llvm::Twine(getName()).str(),
                                      type->clone());
  }

private:
  TypeExpr *getType() { return type.get(); }

  /// LLVM style RTTI
  static bool classof(const Decl *c) { return c->getKind() == Decl_Port; }

private:
  std::unique_ptr<TypeExpr> type;
};

class VariantDecl : public Decl {
public:
  VariantDecl(Location location, std::string name,
              std::vector<std::unique_ptr<FieldDecl>> fields)
      : Decl(Decl_Variant, location, name), fields(std::move(fields)) {}

  std::unique_ptr<Decl> clone() const override {
    return std::unique_ptr<Decl>();
  }
  /// LLVM style RTTI
  static bool classof(const Decl *c) { return c->getKind() == Decl_Variant; }

  llvm::ArrayRef<std::unique_ptr<FieldDecl>> getFields() { return fields; }

private:
  std::vector<std::unique_ptr<FieldDecl>> fields;
};

class AlgebraicTypeDecl {
public:
  virtual ~AlgebraicTypeDecl() = default;

  virtual llvm::ArrayRef<std::unique_ptr<ParameterTypeDecl>>
  getTypeParameters() = 0;
  virtual llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>>
  getValueParameters() = 0;
};

class SumTypeDecl : public GlobalTypeDecl, AlgebraicTypeDecl {
public:
  SumTypeDecl(Location location, std::string name, Availability availability,
              std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
              std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
              std::vector<std::unique_ptr<VariantDecl>> variants)
      : GlobalTypeDecl(location, name, availability),
        typeParameters(std::move(typeParameters)),
        valueParameters(std::move(valueParameters)),
        variants(std::move(variants)) {}

  std::unique_ptr<Decl> clone() const override {
    return std::unique_ptr<Decl>();
  }

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

class ProductTypeDecl : public GlobalTypeDecl, AlgebraicTypeDecl {
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

  std::unique_ptr<Decl> clone() const override {
    return std::unique_ptr<Decl>();
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

class SingleImport : public Import {
public:
  SingleImport(Location location, Prefix prefix,
               std::unique_ptr<QID> globalName, std::string localName)
      : Import(Import_Single, location, prefix),
        globalName(std::move(globalName)), localName(localName) {}

private:
  std::unique_ptr<QID> globalName;
  std::string localName;
};

class GroupImport : public Import {
public:
  GroupImport(Location location, Prefix prefix, std::unique_ptr<QID> globalName)
      : Import(Import_Group, location, prefix),
        globalName(std::move(globalName)) {}

private:
  std::unique_ptr<QID> globalName;
};

/// classes for all type TypeExpr nodes.

class NominalTypeExpr : public TypeExpr {
public:
  NominalTypeExpr(Location loc, std::string name,
                  std::vector<std::unique_ptr<TypeParameter>> parameterType,
                  std::vector<std::unique_ptr<ValueParameter>> parameterValue)
      : TypeExpr(TypeExpr_Nominal, loc), name(name),
        parameterType(std::move(parameterType)),
        parameterValue(std::move(parameterValue)) {}

  std::unique_ptr<TypeExpr> clone() const override {
    std::vector<std::unique_ptr<TypeParameter>> toParameterType;
    toParameterType.reserve(parameterType.size());
    for (const auto &e : parameterType) {
      auto t = std::unique_ptr<cal::TypeParameter>(
          static_cast<cal::TypeParameter *>(e->clone().release()));
      toParameterType.push_back(std::move(t));
    }

    std::vector<std::unique_ptr<ValueParameter>> toParameterValue;
    toParameterType.reserve(parameterValue.size());
    for (const auto &e : parameterValue) {
      auto t = std::unique_ptr<cal::ValueParameter>(
          static_cast<cal::ValueParameter *>(e->clone().release()));
      toParameterValue.push_back(std::move(t));
    }

    return std::make_unique<NominalTypeExpr>(
        loc(), name, std::move(toParameterType), std::move(toParameterValue));
  }

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

class FunctionTypeExpr : public TypeExpr {
public:
  FunctionTypeExpr(Location loc,
                   std::vector<std::unique_ptr<TypeExpr>> parameterTypes,
                   std::unique_ptr<TypeExpr> returnType)
      : TypeExpr(TypeExpr_Function, loc),
        parameterTypes(std::move(parameterTypes)),
        returnType(std::move(returnType)) {}

  std::unique_ptr<TypeExpr> clone() const override {
    std::vector<std::unique_ptr<TypeExpr>> to;
    to.reserve(parameterTypes.size());
    for (const auto &e : parameterTypes) {
      to.push_back(e->clone());
    }
    return std::make_unique<FunctionTypeExpr>(loc(), std::move(to),
                                              returnType->clone());
  }

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

class ProcedureTypeExpr : public TypeExpr {
public:
  ProcedureTypeExpr(Location loc,
                    std::vector<std::unique_ptr<TypeExpr>> parameterTypes)
      : TypeExpr(TypeExpr_Procedure, loc),
        parameterTypes(std::move(parameterTypes)) {}

  std::unique_ptr<TypeExpr> clone() const override {
    std::vector<std::unique_ptr<TypeExpr>> to;
    to.reserve(parameterTypes.size());
    for (const auto &e : parameterTypes) {
      to.push_back(e->clone());
    }
    return std::make_unique<ProcedureTypeExpr>(loc(), std::move(to));
  }

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

class TupleTypeExpr : public TypeExpr {
public:
  TupleTypeExpr(Location loc, std::vector<std::unique_ptr<TypeExpr>> types)
      : TypeExpr(TypeExpr_Tuple, loc), types(std::move(types)) {}

  std::unique_ptr<TypeExpr> clone() const override {
    std::vector<std::unique_ptr<TypeExpr>> to;
    to.reserve(types.size());
    for (const auto &e : types) {
      to.push_back(e->clone());
    }
    return std::make_unique<TupleTypeExpr>(loc(), std::move(to));
  }

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getTypes() { return types; }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Tuple;
  }

private:
  std::vector<std::unique_ptr<TypeExpr>> types;
};

class UnionTypeExpr : public TypeExpr {
public:
  UnionTypeExpr(Location location, std::vector<std::unique_ptr<TypeExpr>> types)
      : TypeExpr(TypeExpr_Union, location), types(std::move(types)) {}

  std::unique_ptr<TypeExpr> clone() const override {
    std::vector<std::unique_ptr<TypeExpr>> to;
    to.reserve(types.size());
    for (const auto &e : types) {
      to.push_back(e->clone());
    }
    return std::make_unique<UnionTypeExpr>(loc(), std::move(to));
  }

  llvm::ArrayRef<std::unique_ptr<TypeExpr>> getTypes() { return types; }

  /// LLVM style RTTI
  static bool classof(const TypeExpr *c) {
    return c->getKind() == TypeExpr_Union;
  }

private:
  std::vector<std::unique_ptr<TypeExpr>> types;
};

/// classes of all expressions nodes.

class ExprLiteralLong : public Expression {
public:
  ExprLiteralLong(Location loc, const long value)
      : Expression(Expr_Literal_Long, loc), value(value) {}
  long getValue() { return value; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralLong(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Literal_Long;
  }

private:
  const long value;
};

class ExprLiteralReal : public Expression {
public:
  ExprLiteralReal(Location loc, const double value)
      : Expression(Expr_Literal_Real, loc), value(value) {}
  double getValue() { return value; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralReal(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Literal_Real;
  }

private:
  const double value;
};

class ExprLiteralChar : public Expression {
public:
  ExprLiteralChar(Location loc, const char value)
      : Expression(Expr_Literal_Char, loc), value(value) {}
  char getValue() { return value; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralChar(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Literal_Char;
  }

private:
  const char value;
};

class ExprLiteralString : public Expression {
public:
  ExprLiteralString(Location loc, std::string value)
      : Expression(Expr_Literal_String, loc), value(value) {}
  llvm::StringRef getValue() { return value; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralString(*this));
  }

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

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralBool(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Literal_Bool;
  }

private:
  const bool value;
};

class ExprLiteralNull : public Expression {
public:
  ExprLiteralNull(Location loc) : Expression(Expr_Literal_Bool, loc) {}

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprLiteralNull(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Literal_Bool;
  }
};

class ExprVariable : public Expression {
public:
  ExprVariable(Location loc, llvm::StringRef name)
      : Expression(Expr_Variable, loc), name(name) {}
  llvm::StringRef getName() { return name; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>(new ExprVariable(*this));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Variable;
  }

private:
  std::string name;
};

class ExprUnary : public Expression {
public:
  ExprUnary(Location loc, std::string Op,
            std::unique_ptr<Expression> expression)
      : Expression(Expr_Unary, loc), op(Op), expression(std::move(expression)) {
  }

  llvm::StringRef getOp() { return op; }
  Expression *getExpression() { return expression.get(); }

  std::unique_ptr<Expression> clone() const override {
    return std::make_unique<ExprUnary>(loc(), op, expression->clone());
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Unary;
  }

private:
  std::string op;
  std::unique_ptr<Expression> expression;
};

class ExprBinary : public Expression {
public:
  llvm::StringRef getOp() { return op; }
  Expression *getLHS() { return lhs.get(); }
  Expression *getRHS() { return rhs.get(); }

  ExprBinary(Location loc, std::string op, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : Expression(Expr_Binary, loc), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  std::unique_ptr<Expression> clone() const override {
    return std::make_unique<ExprBinary>(loc(), op, lhs->clone(), rhs->clone());
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Binary;
  }

private:
  std::string op;
  std::unique_ptr<Expression> lhs, rhs;
};

class ExprComprehension : public Expression {
public:
  ExprComprehension(Location location, std::unique_ptr<Generator> generator,
                    std::vector<std::unique_ptr<Expression>> filters,
                    std::unique_ptr<Expression> collection)
      : Expression(Expr_Comprehension, location),
        generator(std::move(generator)), filters(std::move(filters)),
        collection(std::move(collection)) {}

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<Expression>> toFilters;
    toFilters.reserve(filters.size());
    for (const auto &e : filters) {
      toFilters.push_back(e->clone());
    }
    return std::make_unique<ExprComprehension>(
        loc(), generator->clone(), std::move(toFilters), collection->clone());
  }

  Generator *getGenerator() { return generator.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getFilters() { return filters; }
  Expression *getCollection() { return collection.get(); }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Comprehension;
  }

private:
  std::unique_ptr<Generator> generator;
  std::vector<std::unique_ptr<Expression>> filters;
  std::unique_ptr<Expression> collection;
};

class ExprApplication : public Expression {

public:
  ExprApplication(Location loc, std::string callee,
                  std::vector<std::unique_ptr<Expression>> args)
      : Expression(Expr_Application, loc), callee(callee),
        args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expression>> getArgs() { return args; }

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<Expression>> to;
    to.reserve(args.size());
    for (const auto &e : args) {
      to.push_back(e->clone());
    }
    return std::make_unique<ExprApplication>(loc(), callee, std::move(to));
  }

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

  std::unique_ptr<Expression> clone() const override {
    return std::make_unique<ExprIndexer>(loc(), structure->clone(),
                                         index->clone());
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Indexer;
  }

private:
  std::unique_ptr<Expression> structure, index;
};

class ExprIf : public Expression {
public:
  ExprIf(Location location, std::unique_ptr<Expression> condition,
         std::unique_ptr<Expression> thenExpr,
         std::unique_ptr<Expression> elseExpr)
      : Expression(Expr_If, location), condition(std::move(condition)),
        thenExpr(std::move(thenExpr)), elseExpr(std::move(elseExpr)) {}

  Expression *getCondition() { return condition.get(); }
  Expression *getThen() { return thenExpr.get(); }
  Expression *getElse() { return elseExpr.get(); }

  std::unique_ptr<Expression> clone() const override {
    return std::make_unique<ExprIf>(loc(), condition->clone(),
                                    thenExpr->clone(), elseExpr->clone());
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_If; }

private:
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Expression> thenExpr;
  std::unique_ptr<Expression> elseExpr;
};

class ExprLambda : public Expression {
public:
  ExprLambda(Location location,
             std::vector<std::unique_ptr<ParameterVarDecl>> valueParams,
             std::unique_ptr<Expression> body,
             std::unique_ptr<TypeExpr> returnTypeExpr)
      : Expression(Expr_Lambda, location), valueParams(std::move(valueParams)),
        body(std::move(body)), returnTypeExpr(std::move(returnTypeExpr)) {}

  llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>> getValueParameters() {
    return valueParams;
  }
  Expression *getBody() { return body.get(); }
  TypeExpr *getReturnTypeExpr() { return returnTypeExpr.get(); }

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<ParameterVarDecl>> toValueParams;
    toValueParams.reserve(valueParams.size());
    for (const auto &e : valueParams) {
      auto t = std::unique_ptr<ParameterVarDecl>(
          static_cast<ParameterVarDecl *>(e->clone().release()));
      toValueParams.push_back(std::move(t));
    }
    return std::make_unique<ExprLambda>(loc(), std::move(toValueParams),
                                        body->clone(), returnTypeExpr->clone());
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_Lambda;
  }

private:
  std::vector<std::unique_ptr<ParameterVarDecl>> valueParams;
  std::unique_ptr<Expression> body;
  std::unique_ptr<TypeExpr> returnTypeExpr;
};

class ExprLet : public Expression {
public:
  ExprLet(Location location, std::vector<std::unique_ptr<TypeDecl>> typeDecls,
          std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
          std::unique_ptr<Expression> body)
      : Expression(Expr_Let, location), typeDecls(std::move(typeDecls)),
        varDecls(std::move(varDecls)), body(std::move(body)) {}

  llvm::ArrayRef<std::unique_ptr<TypeDecl>> getTypeDecls() { return typeDecls; }
  llvm::ArrayRef<std::unique_ptr<LocalVarDecl>> getVarDecls() {
    return varDecls;
  }

  Expression *getBody() { return body.get(); }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>();
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Let; }

private:
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::unique_ptr<Expression> body;
};

class ExprList : public Expression {
public:
  ExprList(Location loc, std::vector<std::unique_ptr<Expression>> elements,
           std::vector<std::unique_ptr<Generator>> generators)
      : Expression(Expr_List, loc), elements(std::move(elements)),
        generators(std::move(generators)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }
  llvm::ArrayRef<std::unique_ptr<Generator>> getGenerators() {
    return generators;
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_List; }

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<Expression>> to;
    to.reserve(elements.size());
    for (const auto &e : elements) {
      to.push_back(e->clone());
    }

    std::vector<std::unique_ptr<Generator>> toGenerator;
    to.reserve(generators.size());
    for (const auto &e : generators) {
      toGenerator.push_back(e->clone());
    }

    return std::make_unique<ExprList>(loc(), std::move(to),
                                      std::move(toGenerator));
  }

private:
  std::vector<std::unique_ptr<Expression>> elements;
  std::vector<std::unique_ptr<Generator>> generators;
};

class ExprProc : public Expression {
public:
  ExprProc(Location location,
           std::vector<std::unique_ptr<ParameterVarDecl>> valueParams,
           std::vector<std::unique_ptr<Statement>> body)
      : Expression(Expr_Proc, location), valueParams(std::move(valueParams)),
        body(std::move(body)) {}

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<ParameterVarDecl>> toValueParams;
    toValueParams.reserve(valueParams.size());
    for (const auto &e : valueParams) {
      auto t = std::unique_ptr<ParameterVarDecl>(
          static_cast<ParameterVarDecl *>(e->clone().release()));
      toValueParams.push_back(std::move(t));
    }

    std::vector<std::unique_ptr<Statement>> toBody;
    toBody.reserve(body.size());
    for (const auto &e : body) {
      toBody.push_back(e->clone());
    }

    return std::make_unique<ExprProc>(loc(), std::move(toValueParams),
                                      std::move(toBody));
  }

  llvm::ArrayRef<std::unique_ptr<ParameterVarDecl>> getValueParameters() {
    return valueParams;
  }
  llvm::ArrayRef<std::unique_ptr<Statement>> getBody() { return body; }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Proc; }

private:
  std::vector<std::unique_ptr<ParameterVarDecl>> valueParams;
  std::vector<std::unique_ptr<Statement>> body;
};

class ExprSet : public Expression {
public:
  ExprSet(Location loc, std::vector<std::unique_ptr<Expression>> elements,
          std::vector<std::unique_ptr<Generator>> generators)
      : Expression(Expr_Set, loc), elements(std::move(elements)),
        generators(std::move(generators)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }
  llvm::ArrayRef<std::unique_ptr<Generator>> getGenerators() {
    return generators;
  }

  std::unique_ptr<Expression> clone() const override {
    std::vector<std::unique_ptr<Expression>> to;
    to.reserve(elements.size());
    for (const auto &e : elements) {
      to.push_back(e->clone());
    }

    std::vector<std::unique_ptr<Generator>> toGenerator;
    to.reserve(generators.size());
    for (const auto &e : generators) {
      toGenerator.push_back(e->clone());
    }

    return std::make_unique<ExprSet>(loc(), std::move(to),
                                     std::move(toGenerator));
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Set; }

private:
  std::vector<std::unique_ptr<Expression>> elements;
  std::vector<std::unique_ptr<Generator>> generators;
};

class ExprMap : public Expression {
public:
  ExprMap(Location location,
          std::vector<std::pair<std::unique_ptr<Expression>,
                                std::unique_ptr<Expression>>>
              mappings,
          std::vector<std::unique_ptr<Generator>> generators)
      : Expression(Expr_Map, location), mappings(std::move(mappings)),
        generators(std::move(generators)) {}

  llvm::ArrayRef<
      std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>>
  getMappings() {
    return mappings;
  }
  llvm::ArrayRef<std::unique_ptr<Generator>> getGenerators() {
    return generators;
  }

  std::unique_ptr<Expression> clone() const override { return nullptr; }

  /// LLVM style RTTI
  static bool classof(const Expression *c) { return c->getKind() == Expr_Map; }

private:
  std::vector<
      std::pair<std::unique_ptr<Expression>, std::unique_ptr<Expression>>>
      mappings;
  std::vector<std::unique_ptr<Generator>> generators;
};

class ExprTuple : public Expression {
public:
  ExprTuple(Location loc, std::vector<std::unique_ptr<Expression>> elements)
      : Expression(Expr_Tuple, loc), elements(std::move(elements)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getElements() { return elements; }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>();
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
      : Expression(Expr_TypeAssertion, loc), expr(std::move(expr)) {}

  Expression *getExpr() { return expr.get(); }
  TypeExpr *getType() { return type.get(); }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>();
  }

  /// LLVM style RTTI
  static bool classof(const Expression *c) {
    return c->getKind() == Expr_TypeAssertion;
  }

private:
  std::unique_ptr<Expression> expr;
  std::unique_ptr<TypeExpr> type;
};

class ExprTypeConstruction : public Expression {
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

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>();
  }

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

class ExprCase : public Expression {
public:
  ExprCase(Location location, std::unique_ptr<Pattern> pattern,
           std::vector<std::unique_ptr<Expression>> guards,
           std::unique_ptr<Expression> expression)
      : Expression(Expr_Case, location), pattern(std::move(pattern)),
        guards(std::move(guards)), expression(std::move(expression)) {}

  Pattern *getPattern() { return pattern.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getGuards() { return guards; }
  Expression *getExpression() { return expression.get(); }

  std::unique_ptr<Expression> clone() const override {
    return std::unique_ptr<Expression>();
  }

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

class LValueVariable : public LValue {
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

class LValueDeref : public LValue {
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

class LValueField : public LValue {
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

class LValueIndexer : public LValue {
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

class PatternAlias : public Pattern {
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

class PatternAlternative : public Pattern {
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
  virtual ~PatternDeclaration() = default;
};

class PatternBinding : public Pattern, public PatternDeclaration {
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

class PatternWildcard : public Pattern, public PatternDeclaration {
public:
  PatternWildcard(Location location) : Pattern(Pattern_Wildcard, location) {}
  PatternVarDecl *getDeclaration() override { return nullptr; }
  /// LLVM style RTTI
  static bool classof(const Pattern *c) {
    return c->getKind() == Pattern_Wildcard;
  }
};

class PatternDeconstruction : public Pattern {
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

class PatternExpression : public Pattern {
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

class PatternList : public Pattern {
public:
  PatternList(Location location, std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(Pattern_List, location), patterns(std::move(patterns)) {}

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() { return patterns; }

  static bool classof(const Pattern *c) { return c->getKind() == Pattern_List; }

private:
  std::vector<std::unique_ptr<Pattern>> patterns;
};

class PatternLiteral : public Pattern {
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

class PatternTuple : public Pattern {
public:
  PatternTuple(Location location,
               std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(Pattern_List, location), patterns(std::move(patterns)) {}

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() { return patterns; }

  static bool classof(const Pattern *c) { return c->getKind() == Pattern_List; }

private:
  std::vector<std::unique_ptr<Pattern>> patterns;
};

class PatternVariable : public Pattern {
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

class StmtAssignment : public Statement {
public:
  StmtAssignment(Location location,
                 std::vector<std::unique_ptr<Annotation>> annotations,
                 std::unique_ptr<LValue> lvalue,
                 std::unique_ptr<Expression> expression)
      : Statement(Stmt_Assignment, location),
        annotations(std::move(annotations)), lvalue(std::move(lvalue)),
        expression(std::move(expression)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtAssignment>();
  }

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

class StmtBlock : public Statement {
public:
  StmtBlock(Location location,
            std::vector<std::unique_ptr<Annotation>> annotations,
            std::vector<std::unique_ptr<TypeDecl>> typeDecls,
            std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
            std::vector<std::unique_ptr<Statement>> statements)
      : Statement(Stmt_Block, location), annotations(std::move(annotations)),
        typeDecls(std::move(typeDecls)), varDecls(std::move(varDecls)),
        statements(std::move(statements)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtBlock>();
  }

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

class StmtCall : public Statement {
public:
  StmtCall(Location location, std::unique_ptr<Expression> procedure,
           std::vector<std::unique_ptr<Expression>> args)
      : Statement(Stmt_Call, location), procedure(std::move(procedure)),
        args(std::move(args)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtCall>();
  }

  Expression *getProcedure() { return procedure.get(); }

  llvm::ArrayRef<std::unique_ptr<Expression>> getArguments() { return args; }

  /// LLVM style RTTI
  static bool classof(const Statement *c) { return c->getKind() == Stmt_Call; }

private:
  std::unique_ptr<Expression> procedure;
  std::vector<std::unique_ptr<Expression>> args;
};

class StmtConsume : public Statement {
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

class StmtForeach : public Statement {
public:
  StmtForeach(Location location,
              std::vector<std::unique_ptr<Annotation>> annotations,
              std::vector<std::unique_ptr<Generator>> generators,
              std::vector<std::unique_ptr<Statement>> body)
      : Statement(Stmt_Foreach, location), annotations(std::move(annotations)),
        generators(std::move(generators)), body(std::move(body)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtForeach>();
  }

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }
  llvm::ArrayRef<std::unique_ptr<Generator>> getGenerators() {
    return generators;
  }
  llvm::ArrayRef<std::unique_ptr<Statement>> getStatement() { return body; }

  /// LLVM style RTTI
  static bool classof(const Statement *c) {
    return c->getKind() == Stmt_Foreach;
  }

private:
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::vector<std::unique_ptr<Generator>> generators;
  std::vector<std::unique_ptr<Statement>> body;
};

class StmtIf : public Statement {
public:
  StmtIf(Location location, std::unique_ptr<Expression> condition,
         std::vector<std::unique_ptr<Statement>> thenBranch,
         std::vector<std::unique_ptr<Statement>> elseBranch)
      : Statement(Stmt_If, location), condition(std::move(condition)),
        thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtIf>();
  }

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

class StmtRead : public Statement {
public:
  StmtRead(Location location, std::unique_ptr<Port> port,
           std::vector<std::unique_ptr<LValue>> lvalues,
           std::unique_ptr<Expression> repeat)
      : Statement(Stmt_Read, location), port(std::move(port)),
        lvalues(std::move(lvalues)), repeat(std::move(repeat)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtRead>();
  }

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

class StmtWhile : public Statement {
public:
  StmtWhile(Location location,
            std::vector<std::unique_ptr<Annotation>> annotations,
            std::unique_ptr<Expression> condition,
            std::vector<std::unique_ptr<Statement>> body)
      : Statement(Stmt_While, location), annotations(std::move(annotations)),
        condition(std::move(condition)), body(std::move(body)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtWhile>();
  }

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

class StmtWrite : public Statement {
public:
  StmtWrite(Location location, std::unique_ptr<Port> port,
            std::vector<std::unique_ptr<Expression>> values,
            std::unique_ptr<Expression> repeat)
      : Statement(Stmt_Write, location), port(std::move(port)),
        values(std::move(values)), repeat(std::move(repeat)) {}

  // TODO : Implement me
  std::unique_ptr<Statement> clone() const override {
    return std::unique_ptr<StmtWrite>();
  }

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

/// Base class for Process Description node.
class ProcessDescription {
public:
  ProcessDescription(Location location,
                     std::vector<std::unique_ptr<Statement>> statements,
                     bool repeated)
      : location(location), statements(std::move(statements)),
        repeated(repeated) {}

  virtual ~ProcessDescription() = default;

  llvm::ArrayRef<std::unique_ptr<Statement>> getStatements() {
    return statements;
  }
  bool getRepeated() const { return repeated; }

  const Location &loc() { return location; }

private:
  Location location;
  std::vector<std::unique_ptr<Statement>> statements;
  const bool repeated;
};

/// Base class for Action node.
class Action {
public:
  Action(Location location,
         std::vector<std::unique_ptr<Annotation>> annotations,
         std::unique_ptr<QID> tag,
         std::vector<std::unique_ptr<InputPattern>> inputPatterns,
         std::vector<std::unique_ptr<OutputExpression>> outputExpressions,
         std::vector<std::unique_ptr<TypeDecl>> typeDecls,
         std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
         std::vector<std::unique_ptr<Expression>> guards,
         std::vector<std::unique_ptr<Statement>> body,
         std::unique_ptr<Expression> delay)
      : location(location), annotations(std::move(annotations)),
        tag(std::move(tag)), inputPatterns(std::move(inputPatterns)),
        outputExpressions(std::move(outputExpressions)),
        typeDecls(std::move(typeDecls)), varDecls(std::move(varDecls)),
        guards(std::move(guards)), body(std::move(body)),
        delay(std::move(delay)) {}

  virtual ~Action() = default;
  const Location &loc() { return location; }

  llvm::ArrayRef<std::unique_ptr<Annotation>> getAnnotations() {
    return annotations;
  }

  QID *getTag() { return tag.get(); }

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
  std::unique_ptr<QID> tag;
  std::vector<std::unique_ptr<InputPattern>> inputPatterns;
  std::vector<std::unique_ptr<OutputExpression>> outputExpressions;
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::vector<std::unique_ptr<Expression>> guards;
  std::vector<std::unique_ptr<Statement>> body;
  std::unique_ptr<Expression> delay;
};

/// Base class for Priority node.
class Priority {
public:
  Priority(Location location, std::unique_ptr<QID> high,
           std::unique_ptr<QID> low)
      : location(location), high(std::move(high)), low(std::move(low)) {}

  virtual ~Priority() = default;

  const Location &loc() { return location; }

  QID *getHigh() { return high.get(); }
  QID *getLow() { return low.get(); }

private:
  Location location;
  std::unique_ptr<QID> high;
  std::unique_ptr<QID> low;
};

/// Base class for Transition node.
class Transition {
public:
  Transition(Location location, std::string sourceState,
             std::string targetState,
             std::vector<std::unique_ptr<QID>> actionTag)
      : location(location), sourceState(sourceState), targetState(targetState),
        actionTag(std::move(actionTag)) {}

  virtual ~Transition() = default;

  const Location &loc() { return location; }
  llvm::StringRef getSourceState() { return sourceState; }
  llvm::StringRef getTargetState() { return targetState; }
  llvm::ArrayRef<std::unique_ptr<QID>> getActionTag() { return actionTag; }

private:
  Location location;
  std::string sourceState;
  std::string targetState;
  std::vector<std::unique_ptr<QID>> actionTag;
};

class Schedule {
public:
  enum ScheduleKind { Schedule_FSM, Schedule_Regexp };
  Schedule(ScheduleKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~Schedule() = default;

  // virtual std::unique_ptr<Schedule> clone() const = 0;

  ScheduleKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const ScheduleKind kind;
  Location location;
};

/// Base class for Schedule FSM node.
class ScheduleFSM : public Schedule {
public:
  ScheduleFSM(Location location, std::string initialState,
              std::vector<std::unique_ptr<Transition>> transitions)
      : Schedule(Schedule_FSM, location), initialState(initialState),
        transitions(std::move(transitions)) {}

  std::string getInitialState() { return initialState; }
  llvm::ArrayRef<std::unique_ptr<Transition>> getTransitions() {
    return transitions;
  }

  /// LLVM style RTTI
  static bool classof(const Schedule *c) {
    return c->getKind() == Schedule_FSM;
  }

private:
  std::string initialState;
  std::vector<std::unique_ptr<Transition>> transitions;
};

/// Base class for all regexp nodes.
class RegExp {
public:
  enum RegExpKind {
    RegExp_Tag,
    RegExp_Unary,
    RegExp_Alt,
    RegExp_Seq,
    RegExp_Opt,
    RegExp_Rep
  };
  RegExp(RegExpKind kind, Location location) : kind(kind), location(location) {}
  virtual ~RegExp() = default;

  RegExpKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const RegExpKind kind;
  Location location;
};

class RegExpTag : public RegExp {
public:
  RegExpTag(Location location, std::unique_ptr<QID> tag)
      : RegExp(RegExp_Tag, location), tag(std::move(tag)) {}

  QID *getTag() { return tag.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Tag; }

private:
  std::unique_ptr<QID> tag;
};

class RegExpUnary : public RegExp {
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

class RegExpOpt : public RegExp {
public:
  RegExpOpt(Location location, std::unique_ptr<RegExp> operand)
      : RegExp(RegExp_Opt, location), operand(std::move(operand)) {}

  RegExp *getOperand() { return operand.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Opt; }

private:
  std::unique_ptr<RegExp> operand;
};

class RegExpRep : public RegExp {
public:
  RegExpRep(Location location, std::unique_ptr<RegExp> operand,
            std::unique_ptr<Expression> min, std::unique_ptr<Expression> max)
      : RegExp(RegExp_Rep, location), operand(std::move(operand)),
        min(std::move(min)), max(std::move(max)) {}

  RegExp *getOperand() { return operand.get(); }

  Expression *getMin() { return min.get(); }

  Expression *getMax() { return max.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Rep; }

private:
  std::unique_ptr<RegExp> operand;
  std::unique_ptr<Expression> min;
  std::unique_ptr<Expression> max;
};

class RegExpAlt : public RegExp {
public:
  RegExpAlt(Location loc, std::unique_ptr<RegExp> lhs,
            std::unique_ptr<RegExp> rhs)
      : RegExp(RegExp_Alt, loc), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  RegExp *getLHS() { return lhs.get(); }
  RegExp *getRHS() { return rhs.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Alt; }

private:
  std::unique_ptr<RegExp> lhs, rhs;
};

class RegExpSeq : public RegExp {
public:
  RegExpSeq(Location loc, std::unique_ptr<RegExp> lhs,
            std::unique_ptr<RegExp> rhs)
      : RegExp(RegExp_Seq, loc), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  RegExp *getLHS() { return lhs.get(); }
  RegExp *getRHS() { return rhs.get(); }

  /// LLVM style RTTI
  static bool classof(const RegExp *c) { return c->getKind() == RegExp_Seq; }

private:
  std::unique_ptr<RegExp> lhs, rhs;
};

class ScheduleRegExp : public Schedule {
public:
  ScheduleRegExp(Location location, std::unique_ptr<RegExp> regexp)
      : Schedule(Schedule_Regexp, location), regexp(std::move(regexp)) {}

  RegExp *getRegExp() { return regexp.get(); }

  /// LLVM style RTTI
  static bool classof(const Schedule *c) {
    return c->getKind() == Schedule_Regexp;
  }

private:
  std::unique_ptr<RegExp> regexp;
};

/// Base class for all Entity nodes.
class Entity {
public:
  enum EntityKind { Entity_Actor, Entity_Network, Entity_AM };

  Entity(EntityKind kind, Location location, std::string name,
         std::vector<std::unique_ptr<Annotation>> annotations,
         std::vector<std::unique_ptr<PortDecl>> inputPorts,
         std::vector<std::unique_ptr<PortDecl>> outputPorts,
         std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
         std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters)
      : kind(kind), location(location), name(name),
        annotations(std::move(annotations)), inputPorts(std::move(inputPorts)),
        outputPorts(std::move(outputPorts)),
        typeParameters(std::move(typeParameters)),
        valueParameters(std::move(valueParameters)) {}

  virtual ~Entity() = default;

  EntityKind getKind() const { return kind; }

  llvm::StringRef getName() { return name; }

  const Location &loc() { return location; }

private:
  const EntityKind kind;
  Location location;
  std::string name;
  std::vector<std::unique_ptr<Annotation>> annotations;
  std::vector<std::unique_ptr<PortDecl>> inputPorts;
  std::vector<std::unique_ptr<PortDecl>> outputPorts;
  std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters;
  std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters;
};

class CalActor : public Entity {
public:
  CalActor(Location location, std::string name,
           std::vector<std::unique_ptr<Annotation>> annotations,
           std::vector<std::unique_ptr<PortDecl>> inputPorts,
           std::vector<std::unique_ptr<PortDecl>> outputPorts,
           std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
           std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
           std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
           std::vector<std::unique_ptr<TypeDecl>> typeDecls,
           std::vector<std::unique_ptr<Expression>> invariants,
           std::vector<std::unique_ptr<Action>> actions,
           std::vector<std::unique_ptr<Action>> initializers,
           std::unique_ptr<ProcessDescription> process,
           std::unique_ptr<Schedule> schedule,
           std::vector<std::unique_ptr<QID>> priorities)
      : Entity(Entity_Actor, location, name, std::move(annotations),
               std::move(inputPorts), std::move(outputPorts),
               std::move(typeParameters), std::move(valueParameters)),
        varDecls(std::move(varDecls)), typeDecls(std::move(typeDecls)),
        invariants(std::move(invariants)), actions(std::move(actions)),
        initializers(std::move(initializers)), process(std::move(process)),
        schedule(std::move(schedule)), priorities(std::move(priorities)) {}

  CalActor(Location location, std::string name,
           std::vector<std::unique_ptr<PortDecl>> inputPorts,
           std::vector<std::unique_ptr<PortDecl>> outputPorts,
           std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
           std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters)
      : CalActor(location, name, std::vector<std::unique_ptr<Annotation>>(),
                 std::move(inputPorts), std::move(outputPorts),
                 std::move(typeParameters), std::move(valueParameters),
                 std::vector<std::unique_ptr<LocalVarDecl>>(),
                 std::vector<std::unique_ptr<TypeDecl>>(),
                 std::vector<std::unique_ptr<Expression>>(),
                 std::vector<std::unique_ptr<Action>>(),
                 std::vector<std::unique_ptr<Action>>(),
                 std::unique_ptr<ProcessDescription>(),
                 std::unique_ptr<Schedule>(),
                 std::vector<std::unique_ptr<QID>>()) {}

  llvm::ArrayRef<std::unique_ptr<LocalVarDecl>> getVarDecls() {
    return varDecls;
  }
  llvm::ArrayRef<std::unique_ptr<TypeDecl>> getTypeDecls() { return typeDecls; }
  llvm::ArrayRef<std::unique_ptr<Expression>> getInvariants() {
    return invariants;
  }
  llvm::ArrayRef<std::unique_ptr<Action>> getActions() { return actions; }
  llvm::ArrayRef<std::unique_ptr<Action>> getInitializers() {
    return initializers;
  }
  Schedule *getSchedule() { return schedule.get(); }

  ProcessDescription *getProcess() { return process.get(); }

  llvm::ArrayRef<std::unique_ptr<QID>> getPriorities() { return priorities; }

  void addVarDecl(std::unique_ptr<LocalVarDecl> varDecl) {
    varDecls.push_back(std::move(varDecl));
  }

  void addAction(std::unique_ptr<Action> action) {
    actions.push_back(std::move(action));
  }

  void addInitializer(std::unique_ptr<Action> action) {
    initializers.push_back(std::move(action));
  }

  void addInvariant(std::unique_ptr<Expression> expression) {
    invariants.push_back(std::move(expression));
  }

  void setSchedule(std::unique_ptr<Schedule> schedule_) {
    schedule = std::move(schedule_);
  }

  void setProcess(std::unique_ptr<ProcessDescription> process_) {
    process = std::move(process_);
  }

  void setPriorities(std::vector<std::unique_ptr<QID>> priorities_) {
    priorities = std::move(priorities_);
  }

  /// LLVM style RTTI
  static bool classof(const Entity *c) { return c->getKind() == Entity_Actor; }

private:
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<Expression>> invariants;
  std::vector<std::unique_ptr<Action>> actions;
  std::vector<std::unique_ptr<Action>> initializers;
  std::unique_ptr<ProcessDescription> process;
  std::unique_ptr<Schedule> schedule;
  std::vector<std::unique_ptr<QID>> priorities;
};

/// Base class for all ToolAttribute nodes.
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
  std::string name;
};

class ToolTypeAttribute : public ToolAttribute {
public:
  ToolTypeAttribute(Location location, std::string name,
                    std::unique_ptr<TypeExpr> type)
      : ToolAttribute(ToolAttribute_TYPE, location, name),
        type(std::move(type)) {}

  TypeExpr *getType() { return type.get(); }

  /// LLVM style RTTI
  static bool classof(const ToolAttribute *c) {
    return c->getKind() == ToolAttribute_TYPE;
  }

private:
  std::unique_ptr<TypeExpr> type;
};

class ToolValueAttribute : public ToolAttribute {
public:
  ToolValueAttribute(Location location, std::string name,
                     std::unique_ptr<Expression> value)
      : ToolAttribute(ToolAttribute_TYPE, location, name),
        value(std::move(value)) {}

  Expression *getValue() { return value.get(); }

  /// LLVM style RTTI
  static bool classof(const ToolAttribute *c) {
    return c->getKind() == ToolAttribute_VALUE;
  }

private:
  std::unique_ptr<Expression> value;
};

/// Base class for all Entity Expression nodes.
class EntityReference {
public:
  enum EntityReferenceKind { EntityRef_Local, EntityRef_Global };
  EntityReference(EntityReferenceKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~EntityReference() = default;

  EntityReferenceKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const EntityReferenceKind kind;
  Location location;
};

class EntityRefLocal : public EntityReference {
public:
  EntityRefLocal(Location location, std::string name)
      : EntityReference(EntityRef_Local, location), name(name) {}

  llvm::StringRef getName() { return name; }
  /// LLVM style RTTI
  static bool classof(const EntityReference *c) {
    return c->getKind() == EntityRef_Local;
  }

private:
  std::string name;
};

class EntityRefGlobal : public EntityReference {
public:
  EntityRefGlobal(Location location, std::unique_ptr<QID> globalName)
      : EntityReference(EntityRef_Global, location),
        globalName(std::move(globalName)) {}

  QID *getGlobalName() { return globalName.get(); }
  /// LLVM style RTTI
  static bool classof(const EntityReference *c) {
    return c->getKind() == EntityRef_Global;
  }

private:
  std::unique_ptr<QID> globalName;
};

class Attributable {
public:
  Attributable(std::vector<std::unique_ptr<ToolAttribute>> attributes)
      : attributes(std::move(attributes)) {}

  virtual ~Attributable() = default;

  llvm::ArrayRef<std::unique_ptr<ToolAttribute>> getAttributes() {
    return attributes;
  }

private:
  std::vector<std::unique_ptr<ToolAttribute>> attributes;
};

/// Base class for all Entity Expression nodes.
class EntityExpr {
public:
  enum EntityExprKind {
    EntityExpr_Comprehension,
    EntityExpr_If,
    EntityExpr_Instance,
    EntityExpr_Lists
  };

  EntityExpr(EntityExprKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~EntityExpr() = default;

  EntityExprKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const EntityExprKind kind;
  Location location;
};

class EntityComprehensionExpr : public EntityExpr {
public:
  EntityComprehensionExpr(Location location,
                          std::unique_ptr<Generator> generator,
                          std::vector<std::unique_ptr<Expression>> filters,
                          std::unique_ptr<EntityExpr> collection)
      : EntityExpr(EntityExpr_Comprehension, location),
        generator(std::move(generator)), filters(std::move(filters)),
        collection(std::move(collection)) {}

  Generator *getGenerator() { return generator.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getFilters() { return filters; }
  EntityExpr *getCollection() { return collection.get(); }
  /// LLVM style RTTI
  static bool classof(const EntityExpr *c) {
    return c->getKind() == EntityExpr_Comprehension;
  }

private:
  std::unique_ptr<Generator> generator;
  std::vector<std::unique_ptr<Expression>> filters;
  std::unique_ptr<EntityExpr> collection;
};

class EntityIfExpr : public EntityExpr {
public:
  EntityIfExpr(Location location, std::unique_ptr<Expression> condition,
               std::unique_ptr<EntityExpr> trueEntity,
               std::unique_ptr<EntityExpr> falseEntity)
      : EntityExpr(EntityExpr_If, location), condition(std::move(condition)),
        trueEntity(std::move(trueEntity)), falseEntity(std::move(falseEntity)) {
  }

  Expression *getCondition() { return condition.get(); }
  EntityExpr *getTrueEntity() { return trueEntity.get(); }
  EntityExpr *getFalseEntity() { return falseEntity.get(); }

  /// LLVM style RTTI
  static bool classof(const EntityExpr *c) {
    return c->getKind() == EntityExpr_If;
  }

private:
  std::unique_ptr<Expression> condition;
  std::unique_ptr<EntityExpr> trueEntity;
  std::unique_ptr<EntityExpr> falseEntity;
};

class EntityInstanceExpr : public EntityExpr, Attributable {
public:
  EntityInstanceExpr(
      Location location, std::vector<std::unique_ptr<ToolAttribute>> attributes,
      std::unique_ptr<EntityReference> entity,
      std::vector<std::unique_ptr<TypeParameter>> typeParameters,
      std::vector<std::unique_ptr<ValueParameter>> valueParameters)
      : EntityExpr(EntityExpr_Instance, location),
        Attributable(std::move(attributes)), entity(std::move(entity)),
        typeParameters(std::move(typeParameters)),
        valueParameters(std::move(valueParameters)) {}

  EntityReference *getEntity() { return entity.get(); }
  llvm::ArrayRef<std::unique_ptr<TypeParameter>> getTypeParameters() {
    return typeParameters;
  }
  llvm::ArrayRef<std::unique_ptr<ValueParameter>> getValueParameters() {
    return valueParameters;
  }
  /// LLVM style RTTI
  static bool classof(const EntityExpr *c) {
    return c->getKind() == EntityExpr_Instance;
  }

private:
  std::unique_ptr<EntityReference> entity;
  std::vector<std::unique_ptr<TypeParameter>> typeParameters;
  std::vector<std::unique_ptr<ValueParameter>> valueParameters;
};

class EntityListExpr : public EntityExpr {
public:
  EntityListExpr(Location location,
                 std::vector<std::unique_ptr<EntityExpr>> entityList)
      : EntityExpr(EntityExpr_Lists, location),
        entityList(std::move(entityList)) {}

  llvm::ArrayRef<std::unique_ptr<EntityExpr>> getEntityList() {
    return entityList;
  }

  /// LLVM style RTTI
  static bool classof(const EntityExpr *c) {
    return c->getKind() == EntityExpr_Lists;
  }

private:
  std::vector<std::unique_ptr<EntityExpr>> entityList;
};

/// Base class for Port Reference node.
class PortReference {
public:
  PortReference(Location location, std::string portName, std::string entityName,
                std::vector<std::unique_ptr<Expression>> entityIndex)
      : location(location), portName(portName), entityName(entityName),
        entityIndex(std::move(entityIndex)) {}

  virtual ~PortReference() = default;

  const Location &loc() { return location; }
  llvm::StringRef getPortName() { return portName; }
  llvm::StringRef getEntityName() { return entityName; }
  llvm::ArrayRef<std::unique_ptr<Expression>> getEntityIndex() {
    return entityIndex;
  }

private:
  Location location;
  std::string portName, entityName;
  std::vector<std::unique_ptr<Expression>> entityIndex;
};

/// Base class for all Structure Statement nodes.
class StructureStmt {
public:
  enum StructureStmtKind {
    StructureStmt_Connection,
    StructureStmt_Foreach,
    StructureStmt_If
  };

  StructureStmt(StructureStmtKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~StructureStmt() = default;

  StructureStmtKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const StructureStmtKind kind;
  Location location;
};

class StructureConnectionStmt : public StructureStmt {
public:
  StructureConnectionStmt(Location location, std::unique_ptr<PortReference> src,
                          std::unique_ptr<PortReference> dst)
      : StructureStmt(StructureStmt_Connection, location), src(std::move(src)),
        dst(std::move(dst)) {}

  PortReference *getSource() { return src.get(); }
  PortReference *getDestination() { return dst.get(); }

  /// LLVM style RTTI
  static bool classof(const StructureStmt *c) {
    return c->getKind() == StructureStmt_Connection;
  }

private:
  std::unique_ptr<PortReference> src, dst;
};

class StructureForeachStmt : public StructureStmt {
public:
  StructureForeachStmt(Location location, std::unique_ptr<Generator> generator,
                       std::vector<std::unique_ptr<Expression>> filters,
                       std::vector<std::unique_ptr<Statement>> statements)
      : StructureStmt(StructureStmt_Foreach, location),
        generator(std::move(generator)), filters(std::move(filters)),
        statements(std::move(statements)) {}

  Generator *getGenerator() { return generator.get(); }
  llvm::ArrayRef<std::unique_ptr<Expression>> getFilters() { return filters; }
  llvm::ArrayRef<std::unique_ptr<Statement>> getStatements() {
    return statements;
  }

  /// LLVM style RTTI
  static bool classof(const StructureStmt *c) {
    return c->getKind() == StructureStmt_Foreach;
  }

private:
  std::unique_ptr<Generator> generator;
  std::vector<std::unique_ptr<Expression>> filters;
  std::vector<std::unique_ptr<Statement>> statements;
};

class StructureIfStmt : public StructureStmt {
public:
  StructureIfStmt(Location location, std::unique_ptr<Expression> condition,
                  std::vector<std::unique_ptr<StructureStmt>> trueStmt,
                  std::vector<std::unique_ptr<StructureStmt>> falseStmt)
      : StructureStmt(StructureStmt_If, location),
        condition(std::move(condition)), trueStmt(std::move(trueStmt)),
        falseStmt(std::move(falseStmt)) {}

  Expression *getCondition() { return condition.get(); }

  llvm::ArrayRef<std::unique_ptr<StructureStmt>> getTrueStmt() {
    return trueStmt;
  }

  llvm::ArrayRef<std::unique_ptr<StructureStmt>> getFalseStmt() {
    return falseStmt;
  }

  /// LLVM style RTTI
  static bool classof(const StructureStmt *c) {
    return c->getKind() == StructureStmt_If;
  }

private:
  std::unique_ptr<Expression> condition;
  std::vector<std::unique_ptr<StructureStmt>> trueStmt;
  std::vector<std::unique_ptr<StructureStmt>> falseStmt;
};

class InstanceDecl {
public:
  InstanceDecl(Location location, std::string name,
               std::unique_ptr<EntityExpr> entity)
      : location(location), name(name), entity(std::move(entity)) {}

  virtual ~InstanceDecl() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }
  EntityExpr *getEntity() { return entity.get(); }

private:
  Location location;
  std::string name;
  std::unique_ptr<EntityExpr> entity;
};

class NLNetwork : public Entity {
public:
  NLNetwork(Location location, std::string name,
            std::vector<std::unique_ptr<Annotation>> annotations,
            std::vector<std::unique_ptr<PortDecl>> inputPorts,
            std::vector<std::unique_ptr<PortDecl>> outputPorts,
            std::vector<std::unique_ptr<ParameterTypeDecl>> typeParameters,
            std::vector<std::unique_ptr<ParameterVarDecl>> valueParameters,
            std::vector<std::unique_ptr<TypeDecl>> typeDecls,
            std::vector<std::unique_ptr<LocalVarDecl>> varDecls,
            std::vector<std::unique_ptr<InstanceDecl>> entities,
            std::vector<std::unique_ptr<StructureStmt>> structure)
      : Entity(Entity_Network, location, name, std::move(annotations),
               std::move(inputPorts), std::move(outputPorts),
               std::move(typeParameters), std::move(valueParameters)),
        typeDecls(std::move(typeDecls)), varDecls(std::move(varDecls)),
        entities(std::move(entities)), structure(std::move(structure)) {}

  llvm::ArrayRef<std::unique_ptr<TypeDecl>> getTypeDecls() { return typeDecls; }

  llvm::ArrayRef<std::unique_ptr<LocalVarDecl>> getVarDecls() {
    return varDecls;
  }

  llvm::ArrayRef<std::unique_ptr<InstanceDecl>> getEntities() {
    return entities;
  }

  llvm::ArrayRef<std::unique_ptr<StructureStmt>> getStructure() {
    return structure;
  }

  /// LLVM style RTTI
  static bool classof(const Entity *c) {
    return c->getKind() == Entity_Network;
  }

private:
  std::vector<std::unique_ptr<TypeDecl>> typeDecls;
  std::vector<std::unique_ptr<LocalVarDecl>> varDecls;
  std::vector<std::unique_ptr<InstanceDecl>> entities;
  std::vector<std::unique_ptr<StructureStmt>> structure;
};

class NamespaceDecl {
public:
  NamespaceDecl(Location location, std::unique_ptr<QID> qid,
                std::vector<std::unique_ptr<Import>> imports,
                std::vector<std::unique_ptr<GlobalVarDecl>> varDecls,
                std::vector<std::unique_ptr<GlobalEntityDecl>> entityDecls,
                std::vector<std::unique_ptr<GlobalTypeDecl>> typeDecl)
      : location(location), qid(std::move(qid)), imports(std::move(imports)),
        varDecls(std::move(varDecls)), entityDecls(std::move(entityDecls)),
        typeDecl(std::move(typeDecl)) {}

  NamespaceDecl()
      : NamespaceDecl(Location(), std::unique_ptr<QID>(),
                      std::vector<std::unique_ptr<Import>>(),
                      std::vector<std::unique_ptr<GlobalVarDecl>>(),
                      std::vector<std::unique_ptr<GlobalEntityDecl>>(),
                      std::vector<std::unique_ptr<GlobalTypeDecl>>()) {}

  virtual ~NamespaceDecl() = default;

  const Location &loc() { return location; }
  QID *getQID() { return qid.get(); }

  llvm::ArrayRef<std::unique_ptr<Import>> getImports() { return imports; }
  llvm::ArrayRef<std::unique_ptr<GlobalVarDecl>> getvarDecls() {
    return varDecls;
  }
  llvm::ArrayRef<std::unique_ptr<GlobalEntityDecl>> getEntityDecls() {
    return entityDecls;
  }
  llvm::ArrayRef<std::unique_ptr<GlobalTypeDecl>> getTypeDecls() {
    return typeDecl;
  }

  void setLocation(Location location_) { location = location_; }

  void addImport(std::unique_ptr<Import> import) {
    imports.push_back(std::move(import));
  }

  void addVarDecl(std::unique_ptr<GlobalVarDecl> variable) {
    varDecls.push_back(std::move(variable));
  }

  void addEntityDecl(std::unique_ptr<GlobalEntityDecl> entity) {
    entityDecls.push_back(std::move(entity));
  }

  void addTypeDecl(std::unique_ptr<GlobalTypeDecl> type) {
    typeDecl.push_back(std::move(type));
  }

  void setQID(std::unique_ptr<QID> qid_) { qid = std::move(qid_); }

private:
  Location location;
  std::unique_ptr<QID> qid;
  std::vector<std::unique_ptr<Import>> imports;
  std::vector<std::unique_ptr<GlobalVarDecl>> varDecls;
  std::vector<std::unique_ptr<GlobalEntityDecl>> entityDecls;
  std::vector<std::unique_ptr<GlobalTypeDecl>> typeDecl;
};

} // namespace cal

#endif