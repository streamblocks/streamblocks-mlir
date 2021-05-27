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
%token
    EOF  0  "end of file"
    /* -- Keywords --*/
    ACTION "action"
    ACTOR "actor"
    ALIAS "alias"
    ALL "all"
    AND "and"
    ANY "any"
    AS "as"
    ASSIGN "assign"
    AT "at"
    AT_STAR "at*"
    BEGIN_ "begin"
    CASE "case"
    CHOOSE "choose"
    CONST "const"
    DEFAULT "default"
    DELAY "delay"
    DIV  "div"
    DO "do"
    DOM "dom"
    ELSE "else"
    ELSIF "elsif"
    END "end"
    ENDACTION "endaction"
    ENDASSIGN "endassign"
    ENDACTOR "endactor"
    ENDCASE "endcase"
    ENDBEGIN "endbegin"
    ENDCHOOSE "endchoose"
    ENDFOREACH "endforeach"
    ENDFUNCTION "endfunction"
    ENDIF "endif"
    ENDINITIALIZE "endinitialize"
    ENDINVARIANT "endinvariant"
    ENDLAMBDA "endlambda"
    ENDLET "endlet"
    ENDPRIORITY "endpriority"
    ENDPROC "endproc"
    ENDPROCEDURE "endprocedure"
    ENDSCHEDULE "endschedule"
    ENDWHILE "endwhile"
    ENTITY "entity"
    ENSURE "ensure"
    FALSE "false"
    FOR "for"
    FOREACH "foreach"
    FSM  "fsm"
    FUNCTION "function"
    GUARD "guard"
    IF "if"
    IMPORT "import"
    IN "in"
    INITIALIZE "initialize"
    INVARIANT "invariant"
    LAMBDA "lambda"
    LET "let"
    MAP "map"
    MOD "mod"
    MULTI "multi"
    MUTABLE "mutable"
    NAMESPACE "namespace"
    NOT "not"
    NULL "null"
    OLD "old"
    OF "of"
    OR "or"
    PRIORITY "priority"
    PROC "proc"
    PACKAGE "package"
    PROCEDURE "procedure"
    REGEXP "regexp"
    REPEAT "repeat"
    REQUIRE "require"
    RNG "rng"
    SCHEDULE "schedule"
    TIME "time"
    THEN "then"
    TRUE "true"
    TO "to"
    TYPE "type"
    VAR "var"
    WHILE "while"
    PUBLIC "public"
    PRIVATE "private"
    LOCAL "local"
    NETWORK "network"
    ENTITIES "entities"
    STRUCTURE "structure"
    EXTERNAL "external"

    /* -- Delimiters --*/
    COLON ":"
    COLON_EQUALS ":="
    COMMA ","
    DASH_DASH_GT "-->"
    DASH_GT "->"
    DOT "."
    DOT_DOT ".."
    EQUALS "="
    EQUALS_EQUALS_GT "==>"
    HASH "#"
    LBRACE "{"
    LBRACK "["
    LPAR "("
    LT "<"
    GT ">"
    PLUS "+"
    QMARK "?"
    RBRACE "}"
    RBRACK "]"
    RPAR ")"
    SEMI ";"
    STAR "*"
    UNDER_SCORE "_"
    VBAR "|"
    CINNAMON_BUN "@"
;

%token <std::string> ID
%token <int> NUMBER


/*%printer { yyo << $$; } <*>;*/

%type <std::unique_ptr<cal::QID>> simple_qid qid
%type <std::unique_ptr<cal::NamespaceDecl>> namespace_decl namespace_decl_default
%type <cal::Import::Prefix> import_kind
%type <std::unique_ptr<cal::Import>> import single_import group_import



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

namespace_decl:  "namespace" qid ":" imports "end"
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




%%

void
cal::CalParser::error (const cal::CalParser::location_type& l, const std::string& m)
{
  std::cerr << l << ": " << m << '\n';
}
