%skeleton "lalr1.cc" /* -*- C++ -*- */
%require "3.5.1"
%defines

%define api.namespace {cal}
%define api.parser.class {CalParser}
%define api.token.constructor
%define api.value.automove
%define api.value.type variant


%code requires {
     #include <memory>
     #include <string>
     #include <vector>

     #include "StreamBlocks/AST/AST.h"

     namespace cal {
        class QID;
     }

  // The following definitions is missing when %locations isn't used
  # ifndef YY_NULLPTR
  #  if defined __cplusplus && 201103L <= __cplusplus
  #   define YY_NULLPTR nullptr
  #  else
  #   define YY_NULLPTR 0
  #  endif
  # endif


  class driver;
}

// The parsing context.
%param { driver& drv }

%locations

%define parse.trace
%define parse.error verbose

%code {
# include "driver.h"
}

%define api.token.prefix {TOK_}

%token EOF  0  "end of file"
/* -- Keywords --*/
%token ACTION "action"
%token ACTOR "actor"
%token ALIAS "alias"
%token ALL "all"
%token ANY "any"
%token AS "as"
%token ASSIGN "assign"
%token AT "at"
%token AT_STAR "at*"
%token BEGIN_ "begin"
%token CASE "case"
%token CHOOSE "choose"
%token CONST "const"
%token DEFAULT "default"
%token DELAY "delay"
%token DO "do"
%token ELSE "else"
%token ELSIF "elsif"
%token END "end"
%token ENDACTION "endaction"
%token ENDASSIGN "endassign"
%token ENDACTOR "endactor"
%token ENDCASE "endcase"
%token ENDBEGIN "endbegin"
%token ENDCHOOSE "endchoose"
%token ENDFOREACH "endforeach"
%token ENDFUNCTION "endfunction"
%token ENDIF "endif"
%token ENDINITIALIZE "endinitialize"
%token ENDINVARIANT "endinvariant"
%token ENDLAMBDA "endlambda"
%token ENDLET "endlet"
%token ENDPRIORITY "endpriority"
%token ENDPROC "endproc"
%token ENDPROCEDURE "endprocedure"
%token ENDSCHEDULE "endschedule"
%token ENDWHILE "endwhile"
%token ENTITY "entity"
%token ENSURE "ensure"
%token FALSE "false"
%token FOR "for"
%token FOREACH "foreach"
%token FSM  "fsm"
%token FUNCTION "function"
%token GUARD "guard"
%token IF "if"
%token IMPORT "import"
%token INITIALIZE "initialize"
%token INVARIANT "invariant"
%token LAMBDA "lambda"
%token LET "let"
%token MAP "map"
%token MULTI "multi"
%token MUTABLE "mutable"
%token NAMESPACE "namespace"
%token NULL "null"
%token OLD "old"
%token OF "of"
%token PRIORITY "priority"
%token PROC "proc"
%token PACKAGE "package"
%token PROCEDURE "procedure"
%token REGEXP "regexp"
%token REPEAT "repeat"
%token REQUIRE "require"

%token SCHEDULE "schedule"
%token TIME "time"
%token THEN "then"
%token TRUE "true"
%token TO "to"
%token TYPE "type"
%token VAR "var"
%token WHILE "while"
%token PUBLIC "public"
%token PRIVATE "private"
%token LOCAL "local"
%token NETWORK "network"
%token ENTITIES "entities"
%token STRUCTURE "structure"
%token EXTERNAL "external"

/* -- Delimiters --*/
%token COLON ":"
%token COLON_COLON "::"
%token COLON_EQUALS ":="
%token COMMA ","
%token DASH_DASH_GT "-->"
%token DASH_GT "->"
%token DOT "."

%token EQUALS_EQUALS_GT "==>"
%token LBRACE "{"
%token LBRACK "["
%token LPAR "("
%token RBRACE "}"
%token RBRACK "]"
%token RPAR ")"
%token SEMI ";"
%token UNDER_SCORE "_"
%token CINNAMON_BUN "@"

/* -- Operators -- */
%token <std::string> AND "and"
%token <std::string> AMPERSAND "&"
%token <std::string> AMPERSAND_AMPERSAND "&&"
%token <std::string> CARET "^"
%token <std::string> DIV "div"
%token <std::string> DOM "dom"
%token <std::string> DOT_DOT ".."
%token <std::string> EQUALS "="
%token <std::string> EQUALS_EQUALS "=="
%token <std::string> HASH "#"
%token <std::string> GT ">"
%token <std::string> GT_EQUALS ">="
%token <std::string> GT_GT ">>"
%token <std::string> IN "in"
%token <std::string> LT "<"
%token <std::string> LT_EQUALS "<="
%token <std::string> LT_LT "<<"
%token <std::string> MINUS "-"
%token <std::string> MOD "mod"
%token <std::string> NOT "not"
%token <std::string> NOT_EQUALS "!="
%token <std::string> OR "or"
%token <std::string> PERC "%"
%token <std::string> PLUS "+"
%token <std::string> QMARK "?"
%token <std::string> RNG "rng"
%token <std::string> SLASH "/"
%token <std::string> STAR "*"
%token <std::string> STAR_STAR "**"
%token <std::string> VBAR "|"
%token <std::string> VBAR_VBAR "||"
%token <std::string> TILDE "~"

%token <std::string> ID
%token <long>        LONG
%token <double>      REAL
%token <std::string> STRING
%token <char>        CHAR


/*%printer { yyo << $$; } <*>;*/

%type <std::unique_ptr<QID>> simple_qid qid

%type <std::unique_ptr<NamespaceDecl>> namespace_decl namespace_decl_default

%type <cal::Import::Prefix> import_kind

%type <std::unique_ptr<Import>> import single_import group_import

%type <std::unique_ptr<Expression>> expr var_expr literal_expr binary_expr unary_expr tuple_expr if_expr elsif_expr function_body let_expr list_expr set_expr lambda_expr proc_expr type_assertion_expr application_expr

%type <std::string> unary_op

%type <std::unique_ptr<Parameter>> parameter_assignment type_parameter value_parameter

%type <std::unique_ptr<TypeExpr>> type nominal_type tuple_type lambda_type function_type

%type <Availability> availability

%type <std::unique_ptr<Variable>> variable

%type <std::unique_ptr<Field>> field

%type <std::unique_ptr<LValue>> lvalue lvalue_variable lvalue_field lvalue_indexer

%type <std::unique_ptr<VarDecl>> simple_var_decl variable_var_decl function_var_decl procedure_var_decl

%type <std::unique_ptr<LocalVarDecl>> local_var_decl block_var_decl

%type <std::unique_ptr<GlobalVarDecl>> global_var_decl

%type <std::unique_ptr<ParameterVarDecl>> formal_value_parameter

%type <std::unique_ptr<GeneratorVarDecl>> generator_var_decl

%type <std::unique_ptr<Statement>> stmt assignment_stmt call_stmt block_stmt if_stmt elsif_stmt while_stmt foreach_stmt

%type <std::unique_ptr<StmtForeach>> foreach_header_stmt

%type <std::vector<std::unique_ptr<Statement>>> foreach_body_stmt

%type <std::unique_ptr<Generator>> for_generator

%left ".."
%left "||" "or"
%left "&&" "and"
%left "|"
%left "^"
%left "&"
%left "!=" "==" "="
%left ">=" ">" "<=" "<"
%left ">>" "<<"
%left "-" "+"
%left "div" "%" "mod" "*" "/"
%left "**"

%nonassoc "if"
%nonassoc "else"

%left ","

%%
%start unit;

unit:
     namespace_decl_default
    | namespace_decl
    ;

/* QID */

simple_qid: ID { $$ = cal::QID::of($1); }
          ;

qid : simple_qid
    | qid "." simple_qid
    {
       $$ = $1;
       $$.get()->concat(std::move($3));
    }
    ;

/* Namespace */

namespace_decl: "namespace" qid ":" namespace_decl_default "end"
              ;

namespace_decl_default : imports.opt global_var_decls.opt
                       ;

/*
namespace_decl_default: stmt
                      ;
*/

/* Imports */

import_kind: "var" { $$ = cal::Import::Prefix::VAR;}
           | "type" { $$ = cal::Import::Prefix::TYPE;}
           | "entity" { $$ = cal::Import::Prefix::TYPE;}
           ;

single_import: import_kind qid "=" ID { $$ = std::make_unique<cal::SingleImport>(@$, $1, std::move($2), $4);}
             | qid "=" ID { $$ = std::make_unique<cal::SingleImport>(@$, cal::Import::Prefix::VAR, std::move($1), $3);}
             | import_kind qid {
                                     std::unique_ptr<cal::QID> global = $2;
                                     $$ = std::make_unique<cal::SingleImport>(@$, $1, std::move(global), global.get()->getLast().get()->toString());
                                }
             | qid {
                        std::unique_ptr<cal::QID> global = $1;
                        $$ = std::make_unique<cal::SingleImport>(@$, cal::Import::Prefix::VAR, std::move(global), global.get()->getLast().get()->toString());
                   }

group_import: "all" qid { $$ = std::make_unique<cal::GroupImport>(@$, cal::Import::Prefix::VAR, std::move($2)); }
            | "all" import_kind qid { $$ = std::make_unique<cal::GroupImport>(@$, $2, std::move($3)); }
            ;

import: "import" single_import  ";" { $$ = $2; }
      | "import" group_import   ";" { $$ = $2; }
      ;

%nterm <std::vector<std::unique_ptr<cal::Import>>> imports.opt;
imports.opt: %empty {/* empty */}
           | imports { $$=$1; }
           ;

%nterm <std::vector<std::unique_ptr<cal::Import>>> imports;
imports: import {$$.push_back($1);}
       | imports import { $$=$1; $$.push_back($2); }
       ;



/* Expression */

expr: var_expr
    | application_expr
    | literal_expr
    | unary_expr
    | binary_expr
    | type_assertion_expr
    | tuple_expr
    | "(" expr ")" {$$ = $2;}
    | if_expr
    | lambda_expr
    | let_expr
    | list_expr
    ;

var_expr: ID { $$ = std::make_unique<cal::ExprVariable>(@$, $1); }
        ;

application_expr: ID "(" exprs ")" { $$ = std::make_unique<ExprApplication>(@$, $1, std::move($3));}
                ;

literal_expr: LONG    { $$ = std::make_unique<cal::ExprLiteralLong>(@$, $1); }
            | REAL    { $$ = std::make_unique<cal::ExprLiteralReal>(@$, $1); }
            | "true"  { $$ = std::make_unique<cal::ExprLiteralBool>(@$, true); }
            | "false" { $$ = std::make_unique<cal::ExprLiteralBool>(@$, false); }
            | "null"  { $$ = std::make_unique<cal::ExprLiteralNull>(@$); }
            | STRING  { $$ = std::make_unique<cal::ExprLiteralString>(@$, $1); }
            | CHAR    { $$ = std::make_unique<cal::ExprLiteralChar>(@$, $1); }
            ;

binary_expr : expr ".."  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "||"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "or"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "&&"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "and" expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "|"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "^"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "&"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "!="  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "=="  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "="   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr ">="  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr ">"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "<="  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "<"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "<<"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr ">>"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "+"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "-"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "div" expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "/"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "mod" expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "%"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "*"   expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            | expr "**"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
            ;

unary_op: "not" | "~" | "#" | "dom" | "rng" | "-"
        ;

unary_expr: unary_op expr %prec "*" { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          ;



tuple_expr: "("  ")" { $$ = std::make_unique<cal::ExprTuple>(@$, std::vector<std::unique_ptr<Expression>>()); }
          | "(" expr "," ")"
          {
            std::vector<std::unique_ptr<Expression>> single_tuple;
            single_tuple.push_back($2);
            $$ = std::make_unique<cal::ExprTuple>(@$, std::move(single_tuple));
          }
          | "(" expr "," exprs opt_comma ")"
          {
             auto tuples = $4;
             tuples.insert(tuples.begin(), $2);
             $$ = std::make_unique<cal::ExprTuple>(@$, std::move(tuples));
          }
          ;
type_assertion_expr: "(" expr "::" type ")" { $$ = std::make_unique<cal::ExprTypeAssertion>(@$, std::move($2), std::move($4)); }
                   ;

opt_comma: %empty
         | ","
         ;

%nterm <std::vector<std::unique_ptr<cal::Expression>>> exprs;
exprs: expr { $$.push_back($1); }
     | exprs "," expr { $$=$1; $$.push_back($3); }
     ;

if_expr: "if" expr "then" expr elsif_expr "end" {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($5)); }
       | "if" expr "then" expr "else" expr "end"  {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($6)); }
       ;

elsif_expr: "elsif" expr "then" expr elsif_expr {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($5)); }
          | "elsif" expr "then" expr "else" expr {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($6)); }
          ;

lambda_expr: "lambda" "(" formal_value_parameters.opt ")" function_type var_decls.opt function_body "end"
             {
                auto returnType = $5;
                auto formal_parameters = $3;
                auto vars = $6;
                std::unique_ptr<Expression> body = $7;
                if(vars.empty()){
                    $$ = std::make_unique<ExprLambda>(@$, std::move(formal_parameters), std::move(body), std::move(returnType));
                } else {
                    std::unique_ptr<Expression> letExpr = std::make_unique<ExprLet>(@6, std::vector<std::unique_ptr<TypeDecl>>(), std::move(vars), std::move(body));
                    $$ = std::make_unique<ExprLambda>(@$, std::move(formal_parameters), std::move(letExpr), std::move(returnType));
                }
             }
           ;

proc_expr: "proc" "(" formal_value_parameters.opt ")" var_decls.opt "begin" stmts "end"
         {
          std::unique_ptr<StmtBlock> body = std::make_unique<StmtBlock>(@4, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::move($5), std::move($7));
          std::vector<std::unique_ptr<Statement>> stmts;
          stmts.push_back(std::move(body));
          $$ = std::make_unique<ExprProc>(@$, std::move($3), std::move(stmts));
          }
         ;

let_expr: "let" block_var_decls ":" expr "end" { $$ = std::make_unique<ExprLet>(@$,std::vector<std::unique_ptr<TypeDecl>>(), std::move($2), std::move($4));  }
        ;



list_expr: "[" exprs "]"  { $$ = make_unique<ExprList>(@$, std::move($2), std::vector<std::unique_ptr<Generator>>()); }
         | "[" exprs ":" for_generators "]"  { $$ = make_unique<ExprList>(@$, std::move($2), std::move($4)); }
         ;

set_expr: "{" exprs "}"  { $$ = make_unique<ExprSet>(@$, std::move($2), std::vector<std::unique_ptr<Generator>>()); }
        | "{" exprs ":" for_generators "}"  { $$ = make_unique<ExprSet>(@$, std::move($2), std::move($4)); }
        ;


/* Filters are not implemented */
for_generator: "for" generator_var_decls "in" expr {$$ = std::make_unique<Generator>(@$, std::unique_ptr<TypeExpr>(), std::move($2), std::move($4), std::vector<std::unique_ptr<Expression>>());}
             | "for" type generator_var_decls "in" expr {$$ = std::make_unique<Generator>(@$, std::move($2), std::move($3), std::move($5), std::vector<std::unique_ptr<Expression>>());}
             ;


%nterm <std::vector<std::unique_ptr<Generator>>> for_generators;
for_generators: for_generator { $$.push_back($1); }
              | for_generators "," for_generator { $$=$1; $$.push_back($3); }
              ;


/* Types */

type : nominal_type
     | tuple_type
     | lambda_type
     ;

%nterm <std::vector<std::unique_ptr<cal::TypeExpr>>> types_plus;
types_plus: type { $$.push_back($1); }
     | types_plus "," type { $$=$1; $$.push_back($3); }
     ;

%nterm <std::vector<std::unique_ptr<cal::TypeExpr>>> types;
types: %empty { /* empty list */ }
     | types_plus { $$=$1; }
     ;

nominal_type: ID
              {
                std::vector<std::unique_ptr<TypeParameter>> types;
                std::vector<std::unique_ptr<ValueParameter>> values;
                $$ = std::make_unique<NominalTypeExpr>(@$, $1, std::move(types), std::move(values));
              }
              | ID "(" parameter_assignments ")"
              {
                 std::vector<std::unique_ptr<TypeParameter>> types;
                 std::vector<std::unique_ptr<ValueParameter>> values;

                 std::vector<std::unique_ptr<Parameter>> parameters = $3;

                 for (auto& item : parameters) {
                    if(item->getKind() == cal::Parameter::ParameterKind::Param_Type){
                        auto t = std::unique_ptr<cal::TypeParameter>(static_cast<cal::TypeParameter*>(item.release()));
                        types.push_back(std::move(t));
                    }else{
                        auto t = std::unique_ptr<cal::ValueParameter>(static_cast<cal::ValueParameter*>(item.release()));
                        values.push_back(std::move(t));
                    }
                 }

                 $$ = std::make_unique<NominalTypeExpr>(@$, $1, std::move(types), std::move(values));
              }
            ;

tuple_type: "(" types_plus ")"  {$$ = std::make_unique<TupleTypeExpr>(@$, std::move($2)); }
          ;

lambda_type: "[" types  "-->" type "]" {$$ = std::make_unique<FunctionTypeExpr>(@$, std::move($2), std::move($4)); }
           | "[" types  "-->"      "]" {$$ = std::make_unique<ProcedureTypeExpr>(@$, std::move($2)); }
           ;


/* Parameter */

%nterm <std::vector<std::unique_ptr<Parameter>>> parameter_assignments;
parameter_assignments: parameter_assignment { $$.push_back($1); }
                     | parameter_assignments "," parameter_assignment { $$=$1; $$.push_back($3); }
                     ;


parameter_assignment: type_parameter
                    | value_parameter
                     ;

value_parameter: ID "=" expr { $$ = std::make_unique<ValueParameter>(@$, $1, std::move($3)); }
               ;

type_parameter: "type" ":" type {$$ = std::make_unique<TypeParameter>(@$, "type", std::move($3));}
              | ID     ":" type {$$ = std::make_unique<TypeParameter>(@$, $1,     std::move($3));}


/* Variable Declaration */

variable: ID {$$ = std::make_unique<Variable>(@$, $1); }
        ;

field: ID {$$ = std::make_unique<Field>(@$, $1); }
     ;


%nterm <std::vector<std::unique_ptr<ParameterVarDecl>>> formal_value_parameters_list;
formal_value_parameters_list: formal_value_parameter { $$.push_back($1); }
     | formal_value_parameters_list "," formal_value_parameter { $$=$1; $$.push_back($3); }
     ;

%nterm <std::vector<std::unique_ptr<ParameterVarDecl>>> formal_value_parameters.opt;
formal_value_parameters.opt: %empty { /* empty list */ }
     | formal_value_parameters_list { $$=$1; }
     ;

formal_value_parameter: variable_var_decl
                        {
                            auto t = std::unique_ptr<ParameterVarDecl>(static_cast<ParameterVarDecl*>($1.release()));
                            $$ = std::move(t);
                        }
                       ;

availability: "public" {$$ = Availability::PUBLIC; }
            | "private" {$$ = Availability::PRIVATE; }
            | "local" {$$ = Availability::LOCAL; }
            ;


%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> block_var_decls_plus;
block_var_decls_plus: block_var_decl { $$.push_back($1); }
                    | block_var_decls_plus "," block_var_decl { $$=$1; $$.push_back($3); }
                    ;

%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> block_var_decls;
block_var_decls: %empty { /* empty list */ }
               | block_var_decls_plus{ $$=$1; }
               ;

block_var_decl: variable_var_decl
                {
                    auto t = std::unique_ptr<cal::LocalVarDecl>(static_cast<cal::LocalVarDecl*>($1.release()));
                    $$ = std::move(t);
                }
              | function_var_decl
                {
                    auto t = std::unique_ptr<cal::LocalVarDecl>(static_cast<cal::LocalVarDecl*>($1.release()));
                    $$ = std::move(t);
                }
              ;


%nterm <std::vector<std::unique_ptr<GlobalVarDecl>>> global_var_decls_plus;
global_var_decls_plus: global_var_decl { $$.push_back($1); }
                     | global_var_decls_plus "," global_var_decl { $$=$1; $$.push_back($3); }
                     ;

%nterm <std::vector<std::unique_ptr<GlobalVarDecl>>> global_var_decls.opt;
global_var_decls.opt: %empty { /* empty list */ }
                | global_var_decls_plus{ $$=$1; }
                ;

global_var_decl:                         simple_var_decl {$$ = std::make_unique<GlobalVarDecl>(std::move($1), false,  Availability::PUBLIC);}
               |              "external" simple_var_decl {$$ = std::make_unique<GlobalVarDecl>(std::move($2), true, Availability::PUBLIC);}
               | availability            simple_var_decl { $$ = std::make_unique<GlobalVarDecl>(std::move($2), false, $1); }
               | availability "external" simple_var_decl { $$ = std::make_unique<GlobalVarDecl>(std::move($3), true,  $1); }
               ;


%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> local_var_decls_plus;
local_var_decls_plus: local_var_decl { $$.push_back($1); }
                     | local_var_decls_plus "," local_var_decl { $$=$1; $$.push_back($3); }
                     ;

%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> local_var_decls;
local_var_decls: %empty { /* empty list */ }
                | local_var_decls_plus{ $$=$1; }
                ;

local_var_decl: "external" simple_var_decl  { $$ = std::make_unique<LocalVarDecl>(@$, std::move($2), true); }
              |            simple_var_decl  { $$ = std::make_unique<LocalVarDecl>(@$, std::move($1), false); }
              ;

generator_var_decl: ID { $$ = std::make_unique<GeneratorVarDecl>(@$, $1); }
                  ;
%nterm <std::vector<std::unique_ptr<GeneratorVarDecl>>> generator_var_decls;
generator_var_decls: generator_var_decl { $$.push_back($1); }
                     | generator_var_decls "," generator_var_decl { $$=$1; $$.push_back($3); }
                     ;

simple_var_decl: variable_var_decl ";"
               | function_var_decl
               | procedure_var_decl
               ;

variable_var_decl:      ID           { $$ = std::make_unique<VarDecl>(@$, $1, std::unique_ptr<TypeExpr>(), std::unique_ptr<Expression>(), true, false); }
                 |      ID ":=" expr  { $$ = std::make_unique<VarDecl>(@$, $1, std::unique_ptr<TypeExpr>(), std::move($3), false, false); }
                 |      ID "="  expr  { $$ = std::make_unique<VarDecl>(@$, $1, std::unique_ptr<TypeExpr>(), std::move($3), true, false); }
                 | type ID            { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::unique_ptr<Expression>(), true, false); }
                 | type ID ":=" expr  { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::move($4), false, false); }
                 | type ID "="  expr  { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::move($4), true, false); }
                 ;



function_type: %empty { $$ = std::unique_ptr<TypeExpr>(); }
             | "-->" type { $$ = std::move($2); }
             ;

%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> var_decls.opt;
var_decls.opt: %empty {$$ = std::vector<std::unique_ptr<LocalVarDecl>>();}
             | "var" block_var_decls {$$ = $2;}


function_body: %empty { $$ = std::unique_ptr<Expression>();}
             | ":" expr {$$ = $2;}

function_var_decl: "function" ID "(" formal_value_parameters.opt ")" function_type var_decls.opt function_body "end"
                   {
                        std::unique_ptr<TypeExpr> type = $6;
                        std::vector<std::unique_ptr<ParameterVarDecl>> parameters = $4;

                        // Clone function return Type
                        std::unique_ptr<TypeExpr> functionReturnType = type->clone();

                        // Clone parameter Types
                        std::vector<std::unique_ptr<TypeExpr>> parameterTypes;
                        parameterTypes.reserve(parameters.size());
                        for (const auto &e : parameters) {
                            std::unique_ptr<cal::TypeExpr> parameterType = e->getType()->clone();
                            parameterTypes.push_back(std::move(parameterType));
                        }

                        std::unique_ptr<Expression> letExpr = std::make_unique<ExprLet>(@$, std::vector<std::unique_ptr<TypeDecl>>(), std::move($7), std::move($8));
                        auto lambdaExpr = std::make_unique<ExprLambda>(@$, std::move(parameters), std::move(letExpr), std::move(type));

                        std::unique_ptr<FunctionTypeExpr> functionTypeExpr = std::make_unique<FunctionTypeExpr>(@6, std::move(parameterTypes), std::move(functionReturnType));

                        $$ = std::make_unique<VarDecl>(@$, $2, std::move(functionTypeExpr), std::move(lambdaExpr), true, false);
                   }
                 ;

procedure_var_decl: "procedure" ID "(" formal_value_parameters.opt ")" var_decls.opt "begin" stmts "end"
                    {
                        std::vector<std::unique_ptr<ParameterVarDecl>> parameters = $4;
                        // Clone parameter Types
                        std::vector<std::unique_ptr<TypeExpr>> parameterTypes;
                        parameterTypes.reserve(parameters.size());
                        for (const auto &e : parameters) {
                            std::unique_ptr<cal::TypeExpr> parameterType = e->getType()->clone();
                            parameterTypes.push_back(std::move(parameterType));
                        }

                        std::unique_ptr<ProcedureTypeExpr> procedureTypeExpr = std::make_unique<ProcedureTypeExpr>(@4, std::move(parameterTypes));


                        std::unique_ptr<StmtBlock> body = std::make_unique<StmtBlock>(@4, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::move($6), std::move($8));
                        std::vector<std::unique_ptr<Statement>> stmts;
                        stmts.push_back(std::move(body));
                        auto procExpr = std::make_unique<ExprProc>(@$, std::move(parameters), std::move(stmts));

                        $$ = std::make_unique<VarDecl>(@$, $2, std::move(procedureTypeExpr), std::move(procExpr), true, false);
                    }
                  ;

/* LValue */

lvalue: lvalue_variable
      | lvalue_field
      | lvalue_indexer
      ;

lvalue_variable: variable { $$ = std::make_unique<LValueVariable>(@$, std::move($1)); }
               ;

lvalue_field: lvalue "." field { $$ = std::make_unique<LValueField>(@$, std::move($1), std::move($3)); }
            ;

lvalue_indexer: lvalue "[" expr "]" { $$ = std::make_unique<LValueIndexer>(@$, std::move($1), std::move($3)); }
              ;


/* Statements */

stmt: assignment_stmt
    | call_stmt
    | block_stmt
    | if_stmt
    | while_stmt
    ;

%nterm <std::vector<std::unique_ptr<Statement>>> stmts.plus;
stmts.plus: stmt { $$.push_back($1); }
          | stmts.plus stmt { $$=$1; $$.push_back($2); }
          ;

%nterm <std::vector<std::unique_ptr<Statement>>> stmts;
stmts: %empty {/* empty */}
     | stmts.plus { $$=$1; }
     ;

assignment_stmt: lvalue ":=" expr ";" { $$ = std::make_unique<StmtAssignment>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move($1), std::move($3)); }
               ;

call_stmt: var_expr "(" exprs ")" ";" { $$ = std::make_unique<StmtCall>(@$, std::move($1), std::move($3)); }
         ;

block_stmt: "begin" stmts "end"                            { $$ = std::make_unique<StmtBlock>(@$, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::vector<std::unique_ptr<LocalVarDecl>>(), std::move($2));}
          | "begin" "var" block_var_decls "do" stmts "end" { $$ = std::make_unique<StmtBlock>(@$, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::move($3), std::move($5));}

if_stmt: "if" expr "then" stmts elsif_stmt "end"
         {
            std::vector<std::unique_ptr<Statement>> elsif;
            elsif.push_back(std::move($5));
            $$ = std::make_unique<StmtIf>(@$, std::move($2), std::move($4), std::move(elsif));
        }
       | "if" expr "then" stmts "else" stmts "end"  {$$ = std::make_unique<StmtIf>(@$, std::move($2), std::move($4), std::move($6)); }
       ;

elsif_stmt: "elsif" expr "then" stmts elsif_stmt
            {
                std::vector<std::unique_ptr<Statement>> elsif;
                elsif.push_back(std::move($5));
                $$ = std::make_unique<StmtIf>(@$, std::move($2), std::move($4), std::move(elsif));
            }
          | "elsif" expr "then" stmts "else" stmts {$$ = std::make_unique<StmtIf>(@$, std::move($2), std::move($4), std::move($6)); }
          ;

while_stmt: "while" expr "do" stmts "end" { $$ = std::make_unique<StmtWhile>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move($2), std::move($4));}
          | "while" expr "var" block_var_decls "do" stmts "end"
            {
                std::unique_ptr<StmtBlock> body = std::make_unique<StmtBlock>(@3, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::move($4), std::move($6));
                std::vector<std::unique_ptr<Statement>> stmts;
                stmts.push_back(std::move(body));
                $$ = std::make_unique<StmtWhile>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move($2), std::move(stmts));
            }
          ;

/*
foreach_stmt: foreach_header_stmt "end" {$$=std::move($1);}
            ;

foreach_header_stmt: "foreach" generator_var_decls "in" expr  foreach_body_stmt
                   {
                        auto generator = std::make_unique<Generator>(@2, std::unique_ptr<TypeExpr>(), std::move($2), std::move($4));
                        $$ = std::make_unique<StmtForeach>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move(generator), std::vector<std::unique_ptr<Expression>>(), std::move($5));
                   }
                   | "foreach" type generator_var_decls "in" expr foreach_body_stmt
                   {
                        auto generator = std::make_unique<Generator>(@2, std::move($2), std::move($3), std::move($5));
                        $$ = std::make_unique<StmtForeach>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move(generator), std::vector<std::unique_ptr<Expression>>(), std::move($6));
                   }
                   | "foreach" generator_var_decls "in" expr "," exprs foreach_body_stmt
                   {
                       auto generator = std::make_unique<Generator>(@2, std::unique_ptr<TypeExpr>(), std::move($2), std::move($4));
                       $$ = std::make_unique<StmtForeach>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move(generator), std::move($6), std::move($7));
                   }
                   | "foreach" type generator_var_decls "in" expr "," exprs foreach_body_stmt
                   {
                       auto generator = std::make_unique<Generator>(@2, std::move($2), std::move($3), std::move($5));
                       $$ = std::make_unique<StmtForeach>(@$, std::vector<std::unique_ptr<Annotation>>(), std::move(generator), std::move($7), std::move($8));
                   }
                   ;


foreach_body_stmt: "," foreach_header_stmt
                 {
                    std::vector<std::unique_ptr<Statement>> stmts;
                    stmts.push_back(std::move($2));
                    $$ = std::move(stmts);
                 }
                 | "do" stmts { $$ = $2;}
                 | "var" block_var_decls "do" stmts
                 {
                    std::unique_ptr<StmtBlock> body = std::make_unique<StmtBlock>(@$, std::vector<std::unique_ptr<Annotation>>(), std::vector<std::unique_ptr<TypeDecl>>(), std::move($2), std::move($4));
                    std::vector<std::unique_ptr<Statement>> stmts;
                    stmts.push_back(std::move(body));
                    $$ = std::move(stmts);
                 }
                 ;
*/
%%

void
cal::CalParser::error (const cal::CalParser::location_type& l, const std::string& m)
{
  std::cerr << l << ": " << m << '\n';
}
