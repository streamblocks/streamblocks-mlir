%code requires {

# define YYLTYPE_IS_DECLARED 1 /* alert the parser that we have our own definition */

}

%{
	#include <stdio.h>
	#include <stack>
	#include "StreamBlocks/AST/AST.h"

	cal::NamespaceDecl *compilationUnit;



	extern int yylex();
	int yyerror(char const * s );
	#define YYERROR_VERBOSE
	#define YYDEBUG 1

	extern std::stack<std::string> fileNames;

    # define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
          (Current).file_name = fileNames.top();            \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
          (Current).file_name = fileNames.top();            \
        }                                                               \
    while (0)
	
%}

%union{
    std::string *string;
    int token;
}


%start compilation_unit
%debug
%verbose
%locations /* track locations: @n of component N; @$ of entire range */


// Identifier
%token <token> ID


// Keywords
%token <token> TK_action "action"
%token <token> TK_actor "actor" 
%token <token> TK_alias "alias" 
%token <token> TK_all "all"
%token <token> TK_and "and"
%token <token> TK_any "any"
%token <token> TK_as "as"
%token <token> TK_assign "assign"
%token <token> TK_at "at"  
%token <token> TK_at_star "at*"
%token <token> TK_begin "begin" 
%token <token> TK_case "case"
%token <token> TK_choose "choose"
%token <token> TK_const "const"
%token <token> TK_default "default"
%token <token> TK_delay "delay"
%token <token> TK_div  "div" 
%token <token> TK_do "do"
%token <token> TK_dom "dom" 
%token <token> TK_else "else" 
%token <token> TK_elsif "elsif" 
%token <token> TK_end "end" 
%token <token> TK_endaction "endaction"
%token <token> TK_endassign "endassign"
%token <token> TK_endactor "endactor"
%token <token> TK_endcase "endcase"
%token <token> TK_endbegin "endbegin"
%token <token> TK_endchoose "endchoose"
%token <token> TK_endforeach "endforeach"
%token <token> TK_endfunction "endfunction"
%token <token> TK_endif "endif"
%token <token> TK_endinitialize "endinitialize"
%token <token> TK_endinvariant "endinvariant"
%token <token> TK_endlambda "endlambda"
%token <token> TK_endlet "endlet"
%token <token> TK_endpriority "endpriority"
%token <token> TK_endproc "endproc"
%token <token> TK_endprocedure "endprocedure"
%token <token> TK_endschedule "endschedule"
%token <token> TK_endwhile "endwhile"
%token <token> TK_entity "entity"
%token <token> TK_ensure "ensure"
%token <token> TK_false "false" 
%token <token> TK_for "for" 
%token <token> TK_foreach "foreach"
%token <token> TK_fsm  "fsm" 
%token <token> TK_function "function" 
%token <token> TK_guard "guard" 
%token <token> TK_if "if" 
%token <token> TK_import "import"
%token <token> TK_in "in" 
%token <token> TK_initialize "initialize" 
%token <token> TK_invariant "invariant" 
%token <token> TK_lambda "lambda" 
%token <token> TK_let "let" 
%token <token> TK_map "map" 
%token <token> TK_mod "mod" 
%token <token> TK_multi "multi"
%token <token> TK_mutable "mutable" 
%token <token> TK_namespace "namespace"
%token <token> TK_not "not" 
%token <token> TK_null "null"
%token <token> TK_old "old"
%token <token> TK_of "of" 
%token <token> TK_or "or" 
%token <token> TK_priority "priority" 
%token <token> TK_proc "proc"
%token <token> TK_package "package"
%token <token> TK_procedure "procedure"
%token <token> TK_regexp "regexp" 
%token <token> TK_repeat "repeat"
%token <token> TK_require "require"
%token <token> TK_rng "rng"
%token <token> TK_schedule "schedule"
%token <token> TK_time "time"
%token <token> TK_then "then" 
%token <token> TK_true "true"
%token <token> TK_to "to"
%token <token> TK_type "type"
%token <token> TK_var "var"
%token <token> TK_while "while" 
%token <token> TK_public "public"
%token <token> TK_private "private"
%token <token> TK_local "local"
%token <token> TK_network "network"
%token <token> TK_entities "entities"
%token <token> TK_structure "structure" 
%token <token> TK_external "external"

// Delimiters, separators, operators 

%token <token> TK_colon
%token <token> TK_colon_equals
%token <token> TK_comma
%token <token> TK_dash_dash_gt
%token <token> TK_dash_gt
%token <token> TK_dot
%token <token> TK_dot_dot
%token <token> TK_equals
%token <token> TK_equals_equals_gt
%token <token> TK_hash
%token <token> TK_lbrace
%token <token> TK_lbrack
%token <token> TK_lpar
%token <token> TK_lt
%token <token> TK_gt
%token <token> TK_plus
%token <token> TK_qmark
%token <token> TK_rbrace
%token <token> TK_rbrack
%token <token> TK_rpar
%token <token> TK_semi
%token <token> TK_star
%token <token> TK_under_score
%token <token> TK_vbar
%token <token> TK_cinnamon_bun

%%


compilation_unit: /* empty */
                | namespace_decl_contents
                | namespace_decl
                ;

annotation: TK_cinnamon_bun ID
          | TK_cinnamon_bun ID TK_lpar annotation_parameters TK_rpar
          | TK_cinnamon_bun ID TK_lpar TK_rpar
          ;

annotations: /* empty */
           | annotations annotation
           ;

annotation_parameters: annotation_parameter
                    | annotation_parameter TK_comma annotation_parameter
                    ;

annotation_parameter: ID TK_equals expression
                    | expression
                    ;

namespace_decl: TK_namespace qid TK_colon namespace_decl_contents TK_end
              ;

namespace_decl_contents: namespce_decl_content
                       | namespce_decl_content namespce_decl_content
                       ;

namespce_decl_content: import TK_semi
                     | actor_decl
                     | network_decl
                     | global_var_decl
                     ;

name: simple_name
    | qid
    ; 

simple_name: ID
           ;

qid: name TK_dot ID
   ;

availability: TK_public 
            | TK_private 
            | TK_local 
            ;

variable_var_decl: type ID
                 | type ID TK_equals expression
                 | type ID TK_colon_equals expression
                 ;

functional_var_decl: TK_function ID TK_lpar formal_value_parameters_opt TK_rpar TK_end
                   | TK_function ID TK_lpar formal_value_parameters_opt TK_rpar function_body TK_end
                   | TK_function ID TK_lpar formal_value_parameters_opt TK_rpar TK_dash_dash_gt type function_body TK_end
                   ;


function_body: var_decl_block TK_semi expression
             | TK_semi expression
             ;

procedure_var_decl: TK_procedure ID TK_lpar formal_value_parameters_opt TK_rpar TK_end
                  | TK_procedure ID TK_lpar formal_value_parameters_opt TK_rpar procedure_body TK_end
                  ;

procedure_body: var_decl_block 
              | var_decl_block TK_begin statements
              | var_decl_block TK_do statements
              ;


var_decl_block: TK_var block_var_decls
              ;

block_var_decl: annotations variable_var_decl TK_semi
              | annotations functional_var_decl
              | annotations procedure_var_decl
              ;

block_var_decls: block_var_decl
               | block_var_decl TK_comma block_var_decl
               ;

local_var_decl: annotations variable_var_decl TK_semi
              | annotations functional_var_decl
              | annotations procedure_var_decl
              | annotations TK_external variable_var_decl TK_semi
              | annotations TK_external functional_var_decl
              | annotations TK_external procedure_var_decl
              ;


global_var_decl: annotations variable_var_decl TK_semi
               | annotations functional_var_decl
               | annotations procedure_var_decl
               | TK_external annotations variable_var_decl TK_semi
               | TK_external annotations functional_var_decl
               | TK_external annotations procedure_var_decl
               | annotations availability variable_var_decl TK_semi
               | annotations availability functional_var_decl
               | annotations availability procedure_var_decl
               | annotations availability TK_external variable_var_decl TK_semi
               | annotations availability TK_external functional_var_decl
               | annotations availability TK_external procedure_var_decl
               ;

import_kind: TK_var
           | TK_type
           | TK_entity
           ;
   

import: TK_import group_import_tail
      | TK_import single_import_tail
      ;

group_import_tail: TK_all qid
               | TK_all import_kind qid
               ;


single_import_tail: qid
                  | qid TK_equals ID
                  | import_kind qid
                  | import_kind qid TK_equals ID
                  ;

formal_value_parameters_opt: formal_value_parameters
                           | /* empty */
                           ;

formal_value_parameters: formal_value_parameters formal_value_parameter
                       | formal_value_parameter
                       ;

formal_value_parameter: variable_var_decl
                      ;

actor_decl:
;

network_decl:
;



type:
;

expression:
;

statements:
;

%%