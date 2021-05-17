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
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include "Visitor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

struct Location
{
    int from_line{0};
    int from_column{0};
    int to_line{0};
    int to_column{0};
    std::string file_name;
};

namespace cal
{

    /// Base class for all expressions nodes.
    class Expression
    {
    public:
        enum ExpressionKind
        {
            Expr_Literal_Integer,
            Expr_Literal_Double,
            Expr_Literal_String,
            Expr_Literal_Bool,
            Expr_Variable,
            Expr_Unary,
            Expr_Binary,
            Expr_Application
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

    class ExprLiteralInteger : public Expression
    {
    public:
        ExprLiteralInteger(Location loc, std::string text) : Expression(Expr_Literal_Integer, loc), value(value) {}
        int getValue() { return value; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal_Integer; }

    private:
        const int value;
    };

    class ExprLiteralDouble : public Expression
    {
    public:
        ExprLiteralDouble(Location loc, std::string text) : ExprLiteralDouble(Expr_Literal_Double, loc), value(value) {}
        double getValue() { return text; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal_Double; }

    private:
        const double value;
    };

    class ExprLiteralString : public Expression
    {
    public:
        ExprLiteralString(Location loc, std::string text) : ExprLiteralString(Expr_Literal_String, loc), value(value) {}
        std::string getValue() { return text; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal_String; }

    private:
        const std::string value;
    };

    class ExprLiteralBool : public Expression
    {
    public:
        ExprLiteralBool(Location loc, std::string text) : ExprLiteralBool(Expr_Literal_Bool, loc), value(value) {}
        bool getValue() { return text; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal_Bool; }

    private:
        const bool value;
    };

    class ExprVariable : public Expression
    {
    public:
        ExprVariable(Location loc, llvm::StringRef name)
            : Expression(Expr_Variable, loc), name(name) {}
        llvm::StringRef getName() { return name; }

        /// LLVM style RTTI
        static bool classof(const Expression *c) { return c->getKind() == Expr_Variable; }

    private:
        std::string name;
    };

    class ExprUnary : public Expression
    {
    public:
        BinaryExprAST(Location loc, char Op, std::unique_ptr<Expression> expr) : Expression(Expr_Unary, loc), expr(std::move(expr)) {}

        char getOp() { return op; }
        Expression *getExpr() { return expr.get(); }

        /// LLVM style RTTI
        static bool classof(const Expression *c) { return c->getKind() == Expr_Unary; }

    private:
        char op;
        std::unique_ptr<ExprAST> expr;
    };

    class ExprBinary : public Expression
    {
    public:
        char getOp() { return op; }
        Expression *getLHS() { return lhs.get(); }
        Expression *getRHS() { return rhs.get(); }

        ExprBinary(Location loc, char Op, std::unique_ptr<Expression> lhs,
                   std::unique_ptr<Expression> rhs)
            : Expression(Expr_Binary, loc), op(Op), lhs(std::move(lhs)),
              rhs(std::move(rhs)) {}

        /// LLVM style RTTI
        static bool classof(const Expression *c) { return c->getKind() == Expr_Binary; }

    private:
        char op;
        std::unique_ptr<Expression> lhs, rhs;
    };

    class ExprApplication : public Expression
    {

    public:
        ExprApplication(Location loc, const std::string &callee,
                        std::vector<std::unique_ptr<Expression>> args)
            : Expression(Expr_Call, loc), callee(callee), args(std::move(args)) {}

        llvm::StringRef getCallee() { return callee; }
        llvm::ArrayRef<std::unique_ptr<Expression>> getArgs() { return args; }

        /// LLVM style RTTI
        static bool classof(const Expression *c) { return c->getKind() == Expr_Application; }

    private:
        std::string callee;
        std::vector<std::unique_ptr<Expression>> args;
    };

    class ExprList : public Expression
    {
    };

}

#endif