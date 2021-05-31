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
%token COLON_EQUALS ":="
%token COMMA ","
%token DASH_DASH_GT "-->"
%token DASH_GT "->"
%token DOT "."
%token DOT_DOT ".."
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

%type <std::unique_ptr<cal::QID>> simple_qid qid

%type <std::unique_ptr<cal::NamespaceDecl>> namespace_decl namespace_decl_default

%type <cal::Import::Prefix> import_kind

%type <std::unique_ptr<cal::Import>> import single_import group_import

%type <std::unique_ptr<cal::Expression>> expr var_expr literal_expr binary_expr unary_expr tuple_expr if_expr elsif_expr function_body

%type <std::unique_ptr<cal::Parameter>> parameter_assignment type_parameter value_parameter

%type <std::unique_ptr<cal::TypeExpr>> type nominal_type tuple_type lambda_type function_type

%type <std::unique_ptr<VarDecl>> simple_local_var_decl variable_var_decl function_var_decl

%type <std::unique_ptr<LocalVarDecl>> block_var_decl

%type <std::unique_ptr<ParameterVarDecl>> formal_value_parameter

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

%%
%start unit;

unit: %empty
    | namespace_decl_default
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

namespace_decl:  "namespace" qid ":" simple_local_var_decl "end"
              ;

namespace_decl_default : imports
                       ;


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

%nterm <std::vector<std::unique_ptr<cal::Import>>> imports;
imports: %empty {/* empty */}
       | imports import { $$=$1; $$.push_back($2); }



/* Expression */

expr: var_expr
    | literal_expr
    | binary_expr
    | unary_expr
    | tuple_expr
    | "(" expr ")" {$$ = $2;}
    | if_expr
    ;

var_expr : ID { $$ = std::make_unique<cal::ExprVariable>(@$, $1); }
         ;

literal_expr: LONG    { $$ = std::make_unique<cal::ExprLiteralLong>(@$, $1); }
            | REAL    { $$ = std::make_unique<cal::ExprLiteralReal>(@$, $1); }
            | "true"  { $$ = std::make_unique<cal::ExprLiteralBool>(@$, true); }
            | "false" { $$ = std::make_unique<cal::ExprLiteralBool>(@$, false); }
            | "null"  { $$ = std::make_unique<cal::ExprLiteralNull>(@$); }
            | STRING  { $$ = std::make_unique<cal::ExprLiteralString>(@$, $1); }
            | CHAR    { $$ = std::make_unique<cal::ExprLiteralChar>(@$, $1); }
            ;

binary_expr : expr "||"  expr { $$ = std::make_unique<cal::ExprBinary>(@$, $2, std::move($1), std::move($3)); }
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

unary_expr: "not" expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          | "~"   expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          | "#"   expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          | "dom" expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          | "rng" expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          | "-"   expr { $$ = std::make_unique<cal::ExprUnary>(@$, $1, std::move($2)); }
          ;


tuple_expr: "(" tuple_expr_list ")" { $$ = std::make_unique<cal::ExprTuple>(@$, std::move($2)); }

%nterm <std::vector<std::unique_ptr<cal::Expression>>> tuple_expr_list;
tuple_expr_list: expr { $$.push_back($1); }
               | tuple_expr_list "," expr { $$=$1; $$.push_back($3); }
               ;

if_expr: "if" expr "then" expr elsif_expr "end" {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($5)); }
       | "if" expr "then" expr "else" expr "end"  {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($6)); }
       ;

elsif_expr: "elsif" expr "then" expr elsif_expr {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($5)); }
          | "elsif" expr "then" expr "else" expr {$$ = std::make_unique<ExprIf>(@$, std::move($2), std::move($4), std::move($6)); }
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
     | types_plus "," type { $$=$1; $$.push_back($3); }
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


%nterm <std::vector<std::unique_ptr<ParameterVarDecl>>> formal_value_parameters_list;
formal_value_parameters_list: formal_value_parameter { $$.push_back($1); }
     | formal_value_parameters_list "," formal_value_parameter { $$=$1; $$.push_back($3); }
     ;

%nterm <std::vector<std::unique_ptr<ParameterVarDecl>>> formal_value_parameters;
formal_value_parameters: %empty { /* empty list */ }
     | formal_value_parameters_list { $$=$1; }
     ;

formal_value_parameter: variable_var_decl
                        {
                            auto t = std::unique_ptr<ParameterVarDecl>(static_cast<ParameterVarDecl*>($1.release()));
                            $$ = std::move(t);
                        }
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
              ;


simple_local_var_decl: variable_var_decl
                     | function_var_decl
                     ;

variable_var_decl:      ID            { $$ = std::make_unique<VarDecl>(@$, $1, std::move(std::unique_ptr<TypeExpr>()), std::move(std::unique_ptr<Expression>()), true, false); }
                 |      ID "="  expr  { $$ = std::make_unique<VarDecl>(@$, $1, std::move(std::unique_ptr<TypeExpr>()), std::move($3), true, false); }
                 |      ID ":=" expr  { $$ = std::make_unique<VarDecl>(@$, $1, std::move(std::unique_ptr<TypeExpr>()), std::move($3), false, false); }
                 | type ID            { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::move(std::unique_ptr<Expression>()), true, false); }
                 | type ID "="  expr  { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::move($4), true, false); }
                 | type ID ":=" expr  { $$ = std::make_unique<VarDecl>(@$, $2, std::move($1), std::move($4), false, false); }
                 ;



function_type: %empty { $$ = std::unique_ptr<TypeExpr>(); }
             | "-->" type { $$ = std::move($2); }
             ;

%nterm <std::vector<std::unique_ptr<LocalVarDecl>>> function_var_decls;
function_var_decls: %empty {$$ = std::vector<std::unique_ptr<LocalVarDecl>>();}
             | "var" block_var_decls {$$ = $2;}


function_body: %empty { $$ = std::unique_ptr<Expression>();}
             | ":" expr {$$ = $2;}

function_var_decl: "function" ID "(" formal_value_parameters ")" function_type function_var_decls function_body "end"
                   {
                        std::unique_ptr<TypeExpr> type = $6;
                        std::vector<std::unique_ptr<ParameterVarDecl>> parameters = $4;


                        std::unique_ptr<TypeExpr> functionReturnType = $6->clone();
                        std::vector<std::unique_ptr<TypeExpr>> typeParams;
                        typeParams.reserve(parameters.size());
                        // -- TODO
                        for (const auto &e : parameters) {

                        }

                        std::unique_ptr<Expression> letExpr = std::make_unique<ExprLet>(@$, std::vector<std::unique_ptr<TypeDecl>>(), std::move($7), std::move($8));
                        auto lambdaExpr = std::make_unique<ExprLambda>(@$, std::move(parameters), std::move(letExpr), std::move(type));

                        std::unique_ptr<FunctionTypeExpr> functionTypeExpr = std::make_unique<FunctionTypeExpr>(@6, std::move(typeParams), std::move(functionReturnType));

                        $$ = std::make_unique<VarDecl>(@$, $2, std::move(functionTypeExpr), std::move(lambdaExpr), true, false);
                   }
                 ;



%%

void
cal::CalParser::error (const cal::CalParser::location_type& l, const std::string& m)
{
  std::cerr << l << ": " << m << '\n';
}
