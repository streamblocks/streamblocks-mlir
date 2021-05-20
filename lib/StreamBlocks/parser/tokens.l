%{
    #include <string>
    #include <stack>
    #include "StreamBlocks/AST/AST.h"
    #include "parser.hpp"


    #define SAVE_TOKEN yylval.string = new std::string(yytext, yyleng)
    #define SAVE_INTEGER yylval.integer = std::stoll(std::string(yytext, yyleng))
    #define SAVE_NUMBER yylval.number = std::stod(std::string(yytext, yyleng))
    #define SAVE_BOOLEAN yylval.boolean = std::string(yytext, yyleng) == "true" ? 1 : 0
    #define TOKEN(t) (yylval.token = t)

    #ifdef _MSC_VER
    int isatty(int) {return 0;};
    #endif

    extern "C" int yywrap() { return 1; }

    int yycolumn = 1;

    #define YYERROR_VERBOSE 1
    int yyerror(char const * s )
    {
        printf("ERROR %s in '%s' at line %d col %d\n", s, yytext, yylineno, yycolumn);
        printf("  parsed %s %d.%d-%d.%d\n", yylloc.file_name.c_str(), yylloc.first_line, yylloc.first_column, yylloc.last_line, yylloc.last_column);
        return 1;
    }

    std::string g_str;

    /* globals to track current indentation */
    int current_line_indent = 0;   /* indentation of the current line */
    int indent_level = 0;          /* indentation level passed to the parser */
    std::stack<int> curr_indents;
    int first_time = 1 ;

    int parsing_error = 0;

    std::stack<std::string> fileNames;
    std::stack<int> lineNo;
    std::vector<std::string> libPaths;

    #define YY_USER_ACTION do { \
        if( yylloc.last_line < yylineno ) yycolumn = 1 ; \
        yylloc.first_line = yylloc.last_line = yylineno; \
        yylloc.first_column = yycolumn; yylloc.last_column = yycolumn + (int)yyleng - 1; \
        yycolumn += (int)yyleng; \
        yylloc.file_name = fileNames.top(); \
        } while(0) ;


%}
%option yylineno

%x indent

%x BLOCK_COMMENT

DecimalLiteral  [1-9][0-9]*
HexLiteral  0[xX][0-9A-Fa-f]+
OctalLiteral  0[0-7]*
Integer  {DecimalLiteral}|{HexLiteral}|{OctalLiteral}

Exponent [eE][+-]?[0-9]+
Real  (([0-9]+"."[0-9]+{Exponent}?[fFdD]?) | ("."[0-9]+{Exponent}?[fFdD]?) | ([0-9]+{Exponent}[fFdD]?) | ([0-9]+{Exponent}?[fFdD]))

EscapeChar  ("\\"( [ntbrf\\'\"] | ([0-7][0-7]?)|([0-3][0-7][0-7])))
Character  ("'"([^'\\\n\r]|{EscapeChar})"'")
UnterminatedCharacter  ("'"([^'\\\n\r]|{EscapeChar})[\n\r])
String  ("\""([^\"\\\n\r]|{EscapeChar})*"\"")
UnterminatedString  ("\""([^\"\\\n\r]|{EscapeChar})*[\n\r])

Digit [0-9]

Letter [$A-Z_a-z]

SimpleId  ({Letter}({Letter}|{Digit})*)
EscapedId  ("\\"([^\"\\\n\r]|{EscapeChar})*"\\")

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


\r /* cr are ignored */

"action" { return TOKEN(TK_action); }
"actor" { return TOKEN(TK_actor); }
"all" { return TOKEN(TK_all); }
"and" { return TOKEN(TK_and); }
"any" { return TOKEN(TK_any); }
"assign" { return TOKEN(TK_assign); }
"at" { return TOKEN(TK_at); }
"at*" { return TOKEN(TK_at_star); }
"begin" { return TOKEN(TK_begin); }
"case" { return TOKEN(TK_case); }
"const" { return TOKEN(TK_const); }
"choose" { return TOKEN(TK_choose); }
"default" { return TOKEN(TK_default); }
"delay" { return TOKEN(TK_delay); }
"div" { return TOKEN(TK_div); }
"do" { return TOKEN(TK_do); }
"dom" { return TOKEN(TK_dom); }
"else" { return TOKEN(TK_else); }
"elsif" { return TOKEN(TK_elsif); }
"end" { return TOKEN(TK_end); }
"endaction" { return TOKEN(TK_endaction); }
"endactor" { return TOKEN(TK_endactor); }
"endassign" { return TOKEN(TK_endassign); }
"endbegin" { return TOKEN(TK_endbegin); }
"endchoose" { return TOKEN(TK_endchoose); }
"endforeach" { return TOKEN(TK_endforeach); }
"endfunction" { return TOKEN(TK_endfunction); }
"endif" { return TOKEN(TK_endif); }
"endinitialize" { return TOKEN(TK_endinitialize); }
"endinvariant" { return TOKEN(TK_endinvariant); }
"endlambda" { return TOKEN(TK_endlambda); }
"endlet" { return TOKEN(TK_endlet); }
"endpriority" { return TOKEN(TK_endpriority); }
"endproc" { return TOKEN(TK_endproc); }
"endprocedure" { return TOKEN(TK_endprocedure); }
"endschedule" { return TOKEN(TK_endschedule); }
"endwhile" { return TOKEN(TK_endwhile); }
"ensure" { return TOKEN(TK_ensure); }
"false" { return TOKEN(TK_false); }
"for" { return TOKEN(TK_for); }
"foreach" { return TOKEN(TK_foreach); }
"fsm" { return TOKEN(TK_fsm); }
"function" { return TOKEN(TK_function); }
"guard" { return TOKEN(TK_guard); }
"if" { return TOKEN(TK_if); }
"import" { return TOKEN(TK_import); }
"in" { return TOKEN(TK_in); }
"initialize" { return TOKEN(TK_initialize); }
"invariant" { return TOKEN(TK_invariant); }
"lambda" { return TOKEN(TK_lambda); }
"let" { return TOKEN(TK_let); }
"map" { return TOKEN(TK_map); }
"mod" { return TOKEN(TK_mod); }
"multi" { return TOKEN(TK_multi); }
"mutable" { return TOKEN(TK_mutable); }
"not" { return TOKEN(TK_not); }
"null" { return TOKEN(TK_null); }
"old" { return TOKEN(TK_old); }
"or" { return TOKEN(TK_or); }
"package" { return TOKEN(TK_package); }
"priority" { return TOKEN(TK_priority); }
"proc" { return TOKEN(TK_proc); }
"procedure" { return TOKEN(TK_procedure); }
"regexp" { return TOKEN(TK_regexp); }
"repeat" { return TOKEN(TK_repeat); }
"require" { return TOKEN(TK_require); }
"schedule" { return TOKEN(TK_schedule); }
"then" { return TOKEN(TK_then); }
"time" { return TOKEN(TK_time); }
"to" { return TOKEN(TK_to); }
"true" { return TOKEN(TK_true); }
"var" { return TOKEN(TK_var); }
"while" { return TOKEN(TK_while); }

":" { return TOKEN(TK_colon); }
":=" { return TOKEN(TK_colon_equals); }
"," { return TOKEN(TK_comma); }
"-->" { return TOKEN(TK_dash_dash_gt); }
"->" { return TOKEN(TK_dash_gt); }
"." { return TOKEN(TK_dot); }
".." { return TOKEN(TK_dot_dot); }
"=" { return TOKEN(TK_equals); }
"==>" { return TOKEN(TK_equals_equals_gt); }
"#" { return TOKEN(TK_hash); }
"{" { return TOKEN(TK_lbrace); }
"[" { return TOKEN(TK_rbrace); }
"(" { return TOKEN(TK_lpar); }
"<" { return TOKEN(TK_lt); }
">" { return TOKEN(TK_gt); }
"+" { return TOKEN(TK_plus); }
"?" { return TOKEN(TK_qmark); }
"}" { return TOKEN(TK_rbrace); }
"]" { return TOKEN(TK_rbrack); }
")" { return TOKEN(TK_rpar); }
";" { return TOKEN(TK_semi); }
"*" { return TOKEN(TK_star); }
"_" { return TOKEN(TK_under_score); }
"|" { return TOKEN(TK_vbar); }
"@" { return TOKEN(TK_cinnamon_bun); }

. printf("line %d, len %d Unknown token %s !\n", yylineno, yyleng, yytext); yyterminate();

%%