%skeleton "lalr1.cc"
%require  "3.0"
%debug
%defines 
%define api.namespace {cal}
/**
 * bison 3.3.2 change
 * %define parser_class_name to this, updated
 * should work for previous bison versions as 
 * well. -jcb 24 Jan 2020
 */
%define api.parser.class {CalParser}

%code requires{
   namespace cal {
      class CalDriver;
      class CalScanner;
   }

// The following definitions is missing when %locations isn't used
# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

}

%parse-param { CalScanner  &scanner  }
%parse-param { CalDriver  &driver  }

/* verbose error messages */
%define parse.error verbose

%code{
   #include <iostream>
   #include <cstdlib>
   #include <fstream>
   
   /* include for all driver functions */
   #include "CalDriver.h"

#undef yylex
#define yylex scanner.yylex
}

%define api.value.type variant
%locations
%initial-action
{
    // initialize the initial location object
    @$.begin.filename = @$.end.filename = &driver.streamname;
};

%start compilation_unit

// Identifier
%token ID
%token STRING
%token INTEGER
%token REAL

// Keywords
%token  ACTION "action"
%token  ACTOR "actor"
%token  ALIAS "alias"
%token  ALL "all"
%token  AND "and"
%token  ANY "any"
%token  AS "as"
%token  ASSIGN "assign"
%token  AT "at"
%token  AT_STAR "at*"
%token  BEGIN_ "begin"
%token  CASE "case"
%token  CHOOSE "choose"
%token  CONST "const"
%token  DEFAULT "default"
%token  DELAY "delay"
%token  DIV  "div"
%token  DO "do"
%token  DOM "dom"
%token  ELSE "else"
%token  ELSIF "elsif"
%token  END "end"
%token  ENDACTION "endaction"
%token  ENDASSIGN "endassign"
%token  ENDACTOR "endactor"
%token  ENDCASE "endcase"
%token  ENDBEGIN "endbegin"
%token  ENDCHOOSE "endchoose"
%token  ENDFOREACH "endforeach"
%token  ENDFUNCTION "endfunction"
%token  ENDIF "endif"
%token  ENDINITIALIZE "endinitialize"
%token  ENDINVARIANT "endinvariant"
%token  ENDLAMBDA "endlambda"
%token  ENDLET "endlet"
%token  ENDPRIORITY "endpriority"
%token  ENDPROC "endproc"
%token  ENDPROCEDURE "endprocedure"
%token  ENDSCHEDULE "endschedule"
%token  ENDWHILE "endwhile"
%token  ENTITY "entity"
%token  ENSURE "ensure"
%token  FALSE "false"
%token  FOR "for"
%token  FOREACH "foreach"
%token  FSM  "fsm"
%token  FUNCTION "function"
%token  GUARD "guard"
%token  IF "if"
%token  IMPORT "import"
%token  IN "in"
%token  INITIALIZE "initialize"
%token  INVARIANT "invariant"
%token  LAMBDA "lambda"
%token  LET "let"
%token  MAP "map"
%token  MOD "mod"
%token  MULTI "multi"
%token  MUTABLE "mutable"
%token  NAMESPACE "namespace"
%token  NOT "not"
%token  NULL_ "null"
%token  OLD "old"
%token  OF "of"
%token  OR "or"
%token  PRIORITY "priority"
%token  PROC "proc"
%token  PACKAGE "package"
%token  PROCEDURE "procedure"
%token  REGEXP "regexp"
%token  REPEAT "repeat"
%token  REQUIRE "require"
%token  RNG "rng"
%token  SCHEDULE "schedule"
%token  TIME "time"
%token  THEN "then"
%token  TRUE "true"
%token  TO "to"
%token  TYPE "type"
%token  VAR "var"
%token  WHILE "while"
%token  PUBLIC "public"
%token  PRIVATE "private"
%token  LOCAL "local"
%token  NETWORK "network"
%token  ENTITIES "entities"
%token  STRUCTURE "structure"
%token  EXTERNAL "external"

// Delimiters, separators, operators 

%token  COLON
%token  COLON_EQUALS
%token  COMMA
%token  DASH_DASH_GT
%token  DASH_GT
%token  DOT
%token  DOT_DOT
%token  EQUALS
%token  EQUALS_EQUALS_GT
%token  HASH
%token  LBRACE
%token  LBRACK
%token  LPAR
%token  LT
%token  GT
%token  PLUS
%token  QMARK
%token  RBRACE
%token  RBRACK
%token  RPAR
%token  SEMI
%token  STAR
%token  UNDER_SCORE
%token  VBAR
%token  CINNAMON_BUN

%%


compilation_unit: /* empty */
                 namespace_decl
                ;

namespace_decl: NAMESPACE ID COLON END
              ;

%%

void
cal::CalParser::error( const location_type &l, const std::string &err_message )
{
   std::cerr << "Error: " << err_message << " at " << l << "\n";
}
