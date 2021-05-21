%{
/* C++ string header, for string ops below */
#include <string>

/* Implementation of yyFlexScanner */ 
#include "CalScanner.h"

#undef  YY_DECL
#define YY_DECL int cal::CalScanner::yylex( cal::CalParser::semantic_type * const lval, cal::CalParser::location_type *loc )

/* typedef to make the returns for the tokens shorter */
using token = cal::CalParser::token;

/* msvc2010 requires that we exclude this header file. */
#define YY_NO_UNISTD_H

/* update location on matching */
#define YY_USER_ACTION loc->step(); loc->columns(yyleng);

%}

%option debug
%option yyclass="cal::CalScanner"
%option noyywrap
%option yylineno
%option c++


%x indent

%x BLOCK_COMMENT

DecimalLiteral  [1-9][0-9]*
HexLiteral  0[xX][0-9A-Fa-f]+
OctalLiteral  0[0-7]*

dig     [0-9]
num1    [-+]?{dig}+\.([eE][-+]?{dig}+)?
num2    [-+]?{dig}*\.{dig}+([eE][-+]?{dig}+)?
Real  {num1}|{num2}


Digit [0-9]
Letter [$A-Z_a-z]
SimpleId  ({Letter}({Letter}|{Digit})*)

OpLetter  [-!@#$%\^&*/+?~|<=>]
Op  {OpLetter}+


%%

<INITIAL>{
"/*"              BEGIN(BLOCK_COMMENT);
}
<BLOCK_COMMENT>{
"*/"      BEGIN(INITIAL);
[^*\n]+   // eat comment in chunks
"*"       // eat the lone star
\n        yylineno++;
}

"//".*  { /* DO NOTHING */ }




"action" { return( token::ACTION ); }
"actor" { return( token::ACTOR ); }
"all" { return( token::ALL ); }
"and" { return( token::AND ); }
"any" { return( token::ANY ); }
"assign" { return( token::ASSIGN ); }
"at" { return( token::AT ); }
"at*" { return( token::AT_STAR ); }
"begin" { return( token::BEGIN_ ); }
"case" { return( token::CASE ); }
"const" { return( token::CONST ); }
"choose" { return( token::CHOOSE ); }
"default" { return( token::DEFAULT ); }
"delay" { return( token::DELAY ); }
"div" { return( token::DIV ); }
"do" { return( token::DO ); }
"dom" { return( token::DOM ); }
"else" { return( token::ELSE ); }
"elsif" { return( token::ELSIF ); }
"end" { return( token::END ); }
"endaction" { return( token::ENDACTION ); }
"endactor" { return( token::ENDACTOR ); }
"endassign" { return( token::ENDASSIGN ); }
"endbegin" { return( token::ENDBEGIN ); }
"endchoose" { return( token::ENDCHOOSE ); }
"endforeach" { return( token::ENDFOREACH ); }
"endfunction" { return( token::ENDFUNCTION ); }
"endif" { return( token::ENDIF ); }
"endinitialize" { return( token::ENDINITIALIZE ) ; }
"endinvariant" { return( token::ENDINVARIANT ); }
"endlambda" { return( token::ENDLAMBDA ); }
"endlet" { return( token::ENDLET ); }
"endpriority" { return( token::ENDPRIORITY ); }
"endproc" { return( token::ENDPROC ); }
"endprocedure" { return( token::ENDPROCEDURE ); }
"endschedule" { return( token::ENDSCHEDULE ); }
"endwhile" { return( token::ENDWHILE ); }
"ensure" { return( token::ENSURE ); }
"false" { return( token::FALSE ); }
"for" { return( token::FOR ); }
"foreach" { return( token::FOREACH ); }
"fsm" { return( token::FSM ); }
"function" { return( token::FUNCTION ); }
"guard" { return( token::GUARD ); }
"if" { return( token::IF ); }
"import" { return( token::IMPORT ); }
"in" { return( token::IN ); }
"initialize" { return( token::INITIALIZE ); }
"invariant" { return( token::INVARIANT ); }
"lambda" { return( token::LAMBDA ); }
"let" { return( token::LET ); }
"map" { return( token::MAP ); }
"mod" { return( token::MOD ); }
"multi" { return( token::MULTI ); }
"mutable" { return( token::MUTABLE ); }
"namespace" { return( token::NAMESPACE); }
"not" { return( token::NOT ); }
"null" { return( token::NULL_ ); }
"old" { return( token::OLD ); }
"or" { return( token::OR ); }
"package" { return( token::PACKAGE ); }
"priority" { return( token::PRIORITY ); }
"proc" { return( token::PROC ); }
"procedure" { return( token::PROCEDURE ); }
"regexp" { return( token::REGEXP ); }
"repeat" { return( token::REPEAT ); }
"require" { return( token::REQUIRE ); }
"schedule" { return( token::SCHEDULE ); }
"then" { return( token::THEN ); }
"time" { return( token::TIME ); }
"to" { return( token::TO ); }
"true" { return( token::TRUE ); }
"var" { return( token::VAR ); }
"while" { return( token::WHILE ); }

":" { return( token::COLON ); }
":=" { return( token::COLON_EQUALS ); }
"," { return( token::COMMA ); }
"-->" { return( token::DASH_DASH_GT ); }
"->" { return( token::DASH_GT ); }
"." { return( token::DOT ); }
".." { return( token::DOT_DOT ); }
"=" { return( token::EQUALS ); }
"==>" { return( token::EQUALS_EQUALS_GT ); }
"#" { return( token::HASH ); }
"{" { return( token::LBRACE ); }
"[" { return( token::RBRACE ); }
"(" { return( token::LPAR ); }
"<" { return( token::LT ); }
">" { return( token::GT ); }
"+" { return( token::PLUS ); }
"?" { return( token::QMARK ); }
"}" { return( token::RBRACE ); }
"]" { return( token::RBRACK ); }
")" { return( token::RPAR ); }
";" { return( token::SEMI ); }
"*" { return( token::STAR ); }
"_" { return( token::UNDER_SCORE ); }
"|" { return( token::VBAR ); }
"@" { return( token::CINNAMON_BUN); }

{DecimalLiteral}  {}
{HexLiteral}      {}
{OctalLiteral}    {}

{Real}            {}


{SimpleId} { return( token::ID ); }


[ \t\r\n\f]* { /* ignore white space. */ }

. printf("line %d, len %d Unknown token %s !\n", yylineno, yyleng, yytext); yyterminate();

%%