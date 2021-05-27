%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "driver.h"
# include "cal_parser.hpp"
%}

%{
#if defined __clang__
# define CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#endif

// Clang and ICC like to pretend they are GCC.
#if defined __GNUC__ && !defined __clang__ && !defined __ICC
# define GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

// Pacify warnings in yy_init_buffer (observed with Flex 2.6.4)
// and GCC 6.4.0, 7.3.0 with -O3.
#if defined GCC_VERSION && 600 <= GCC_VERSION
# pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

// This example uses Flex's C backend, yet compiles it as C++.
// So expect warnings about C style casts and NULL.
#if defined CLANG_VERSION && 500 <= CLANG_VERSION
# pragma clang diagnostic ignored "-Wold-style-cast"
# pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined GCC_VERSION && 407 <= GCC_VERSION
# pragma GCC diagnostic ignored "-Wold-style-cast"
# pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#define FLEX_VERSION (YY_FLEX_MAJOR_VERSION * 100 + YY_FLEX_MINOR_VERSION)

// Old versions of Flex (2.5.35) generate an incomplete documentation comment.
//
//  In file included from src/scan-code-c.c:3:
//  src/scan-code.c:2198:21: error: empty paragraph passed to '@param' command
//        [-Werror,-Wdocumentation]
//   * @param line_number
//     ~~~~~~~~~~~~~~~~~^
//  1 error generated.
#if FLEX_VERSION < 206 && defined CLANG_VERSION
# pragma clang diagnostic ignored "-Wdocumentation"
#endif

// Old versions of Flex (2.5.35) use 'register'.  Warnings introduced in
// GCC 7 and Clang 6.
#if FLEX_VERSION < 206
# if defined CLANG_VERSION && 600 <= CLANG_VERSION
#  pragma clang diagnostic ignored "-Wdeprecated-register"
# elif defined GCC_VERSION && 700 <= GCC_VERSION
#  pragma GCC diagnostic ignored "-Wregister"
# endif
#endif

#if FLEX_VERSION < 206
# if defined CLANG_VERSION
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wdocumentation"
#  pragma clang diagnostic ignored "-Wshorten-64-to-32"
#  pragma clang diagnostic ignored "-Wsign-conversion"
# elif defined GCC_VERSION
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-conversion"
# endif
#endif
%}

%option noyywrap nounput noinput batch debug

%{
  // A number symbol corresponding to the value in S.
  cal::CalParser::symbol_type
  make_NUMBER (const std::string &s, const cal::CalParser::location_type& loc);
%}

id    [a-zA-Z][a-zA-Z_0-9]*
int   [0-9]+
blank [ \t\r]

%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}

%x BLOCK_COMMENT

%%
%{
  // A handy shortcut to the location held by the driver.
  cal::location& loc = drv.location;
  // Code run each time yylex is called.
  loc.step ();
%}
{blank}+   loc.step ();
\n+        loc.lines (yyleng); loc.step ();

<INITIAL>{
"/*"              BEGIN(BLOCK_COMMENT);
}
<BLOCK_COMMENT>{
"*/"      BEGIN(INITIAL);
[^*\n]+   // eat comment in chunks
"*"       // eat the lone star
\n        loc.lines (yyleng);
}



"action" { return cal::CalParser::make_ACTION(loc); }
"actor" { return cal::CalParser::make_ACTOR(loc); }
"all" { return cal::CalParser::make_ALL(loc); }
"and" { return cal::CalParser::make_AND(loc); }
"any" { return cal::CalParser::make_ANY(loc); }
"assign" { return cal::CalParser::make_ASSIGN(loc); }
"at" { return cal::CalParser::make_AT(loc); }
"at*" { return cal::CalParser::make_AT_STAR(loc); }
"begin" { return cal::CalParser::make_BEGIN_(loc); }
"case" { return cal::CalParser::make_CASE(loc); }
"const" { return cal::CalParser::make_CONST(loc); }
"choose" { return cal::CalParser::make_CHOOSE(loc); }
"default" { return cal::CalParser::make_DEFAULT(loc); }
"delay" { return cal::CalParser::make_DELAY(loc); }
"div" { return cal::CalParser::make_DIV(loc); }
"do" { return cal::CalParser::make_DO(loc); }
"dom" { return cal::CalParser::make_DOM(loc); }
"else" { return cal::CalParser::make_ELSE(loc); }
"elsif" { return cal::CalParser::make_ELSIF(loc); }
"end" { return cal::CalParser::make_END(loc); }
"endaction" { return cal::CalParser::make_ENDACTION(loc); }
"endactor" { return cal::CalParser::make_ENDACTOR(loc); }
"endassign" { return cal::CalParser::make_ENDASSIGN(loc); }
"endbegin" { return cal::CalParser::make_ENDBEGIN(loc); }
"endchoose" { return cal::CalParser::make_ENDCHOOSE(loc); }
"endforeach" { return cal::CalParser::make_ENDFOREACH(loc); }
"endfunction" { return cal::CalParser::make_ENDFUNCTION(loc); }
"endif" { return cal::CalParser::make_ENDIF(loc); }
"endinitialize" { return cal::CalParser::make_ENDINITIALIZE(loc); }
"endinvariant" { return cal::CalParser::make_ENDINVARIANT(loc); }
"endlambda" { return cal::CalParser::make_ENDLAMBDA(loc); }
"endlet" { return cal::CalParser::make_ENDLET(loc); }
"endpriority" { return cal::CalParser::make_ENDPRIORITY(loc); }
"endproc" { return cal::CalParser::make_ENDPROC(loc); }
"endprocedure" { return cal::CalParser::make_ENDPROCEDURE(loc); }
"endschedule" { return cal::CalParser::make_ENDSCHEDULE(loc); }
"endwhile" { return cal::CalParser::make_ENDWHILE(loc); }
"ensure" { return cal::CalParser::make_ENSURE(loc); }
"false" { return cal::CalParser::make_FALSE(loc); }
"for" { return cal::CalParser::make_FOR(loc); }
"foreach" { return cal::CalParser::make_FOREACH(loc); }
"fsm" { return cal::CalParser::make_FSM(loc); }
"function" { return cal::CalParser::make_FUNCTION(loc); }
"guard" { return cal::CalParser::make_GUARD(loc); }
"if" { return cal::CalParser::make_IF(loc); }
"import" { return cal::CalParser::make_IMPORT(loc); }
"in" { return cal::CalParser::make_IN(loc); }
"initialize" { return cal::CalParser::make_INITIALIZE(loc); }
"invariant" { return cal::CalParser::make_INVARIANT(loc); }
"lambda" { return cal::CalParser::make_LAMBDA(loc); }
"let" { return cal::CalParser::make_LET(loc); }
"map" { return cal::CalParser::make_MAP(loc); }
"mod" { return cal::CalParser::make_MOD(loc); }
"multi" { return cal::CalParser::make_MULTI(loc); }
"mutable" { return cal::CalParser::make_MUTABLE(loc); }
"namespace" { return cal::CalParser::make_NAMESPACE(loc); }
"not" { return cal::CalParser::make_NOT(loc); }
"null" { return cal::CalParser::make_NULL(loc); }
"old" { return cal::CalParser::make_OLD(loc); }
"or" { return cal::CalParser::make_OR(loc); }
"package" { return cal::CalParser::make_PACKAGE(loc); }
"priority" { return cal::CalParser::make_PRIORITY(loc); }
"proc" { return cal::CalParser::make_PROC(loc); }
"procedure" { return cal::CalParser::make_PROCEDURE(loc); }
"regexp" { return cal::CalParser::make_REGEXP(loc); }
"repeat" { return cal::CalParser::make_REPEAT(loc); }
"require" { return cal::CalParser::make_REQUIRE(loc); }
"schedule" { return cal::CalParser::make_SCHEDULE(loc); }
"then" { return cal::CalParser::make_THEN(loc); }
"time" { return cal::CalParser::make_TIME(loc); }
"to" { return cal::CalParser::make_TO(loc); }
"true" { return cal::CalParser::make_TRUE(loc); }
"var" { return cal::CalParser::make_VAR(loc); }
"while" { return cal::CalParser::make_WHILE(loc); }

":" { return cal::CalParser::make_COLON(loc); }
":=" { return cal::CalParser::make_COLON_EQUALS(loc); }
"," { return cal::CalParser::make_COMMA(loc); }
"-->" { return cal::CalParser::make_DASH_DASH_GT(loc); }
"->" { return cal::CalParser::make_DASH_GT(loc); }
"." { return cal::CalParser::make_DOT(loc); }
".." { return cal::CalParser::make_DOT_DOT(loc); }
"=" { return cal::CalParser::make_EQUALS(loc); }
"==>" { return cal::CalParser::make_EQUALS_EQUALS_GT(loc); }
"#" { return cal::CalParser::make_HASH(loc); }
"{" { return cal::CalParser::make_LBRACE(loc); }
"[" { return cal::CalParser::make_RBRACE(loc); }
"(" { return cal::CalParser::make_LPAR(loc); }
"<" { return cal::CalParser::make_LT(loc); }
">" { return cal::CalParser::make_GT(loc); }
"+" { return cal::CalParser::make_PLUS(loc); }
"?" { return cal::CalParser::make_QMARK(loc); }
"}" { return cal::CalParser::make_RBRACE(loc); }
"]" { return cal::CalParser::make_RBRACK(loc); }
")" { return cal::CalParser::make_RPAR(loc); }
";" { return cal::CalParser::make_SEMI(loc); }
"*" { return cal::CalParser::make_STAR(loc); }
"_" { return cal::CalParser::make_UNDER_SCORE(loc); }
"|" { return cal::CalParser::make_VBAR(loc); }
"@" { return cal::CalParser::make_CINNAMON_BUN(loc); }





{int}      return make_NUMBER (yytext, loc);
{id}       return cal::CalParser::make_ID (yytext, loc);
.          {
             throw cal::CalParser::syntax_error
               (loc, "invalid character: " + std::string(yytext));
}
<<EOF>>    return cal::CalParser::make_EOF (loc);
%%

cal::CalParser::symbol_type
make_NUMBER (const std::string &s, const cal::CalParser::location_type& loc)
{
  errno = 0;
  long n = strtol (s.c_str(), NULL, 10);
  if (! (INT_MIN <= n && n <= INT_MAX && errno != ERANGE))
    throw cal::CalParser::syntax_error (loc, "integer is out of range: " + s);
  return cal::CalParser::make_NUMBER ((int) n, loc);
}

void
driver::scan_begin ()
{
  yy_flex_debug = trace_scanning;
  if (file.empty () || file == "-")
    yyin = stdin;
  else if (!(yyin = fopen (file.c_str (), "r")))
    {
      std::cerr << "cannot open " << file << ": " << strerror(errno) << '\n';
      exit (EXIT_FAILURE);
    }
}

void
driver::scan_end ()
{
  fclose (yyin);
}
