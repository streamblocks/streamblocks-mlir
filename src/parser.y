{    
    #include <cstdio>
    #include <string>
    #include <iostream>
    
    using namespace std;
    #define YYSTYPE string
    #define YYERROR_VERBOSE 1
    #define DEBUG
    
    int  wrapRet = 1;
    
    int yylex(void);
    extern "C" {
        int yywrap( void ) {
            return wrapRet;
        }
    }
    void yyerror(const char *str) {
        #ifdef DEBUG
          //cout << "CAL Parser: " << str << endl;
        #endif
    }

    int main();

%}

// Identifier
%token ID


// Keywords
%token TK_action "action"
%token TK_actor "actor" 
%token TK_alias "alias" 
%token TK_all "all"
%token TK_and "and" 
%token TK_as "as"  
%token TK_begin "begin" 
%token TK_case "case"
%token TK_const "const" 
%token TK_div  "div" 
%token TK_do "do"
%token TK_dom "dom" 
%token TK_else "else" 
%token TK_elsif "elsif" 
%token TK_end "end" 
%token TK_end_action "endaction" 
%token TK_end_actor "endactor" 
%token TK_end_case "endcase" 
%token TK_end_choose "endchoose" 
%token TK_end_foreach "endforeach" 
%token TK_end_function "endfunction"
%token TK_end_if "endif" 
%token TK_end_initialize "endinitialize" 
%token TK_end_invariant "endinvariant"
%token TK_end_lambda "endlambda"    
%token TK_end_let "endlet" 
%token TK_end_priority "endpriority" 
%token TK_end_proc "endproc" 
%token TK_end_procedure "endprocedure" 
%token TK_end_schedule "endschedule" 
%token TK_end_while "endwhile" 
%token TK_entity "entity"
%token TK_false "false" 
%token TK_for "for" 
%token TK_foreach "foreach"
%token TK_fsm  "fsm" 
%token TK_function "function" 
%token TK_guard "guard" 
%token TK_if "if" 
%token TK_import "import"
%token TK_in "in" 
%token TK_initialize "initialize" 
%token TK_invariant "invariant" 
%token TK_lambda "lambda" 
%token TK_let "let" 
%token TK_map "map" 
%token TK_mod "mod" 
%token TK_multi "multi"
%token TK_mutable "mutable" 
%token TK_namespace "namespace"
%token TK_not "not" 
%token TK_null "null"
%token TK_old "old"
%token TK_of "of" 
%token TK_or "or" 
%token TK_priority "priority" 
%token TK_proc "proc" 
%token TK_procedure "procedure" 
%token TK_regexp "regexp" 
%token TK_repeat "repeat" 
%token TK_rng "rng"
%token TK_schedule "schedule" 
%token TK_then "then" 
%token TK_true "true" 
%token TK_type "type"
%token TK_var "var"
%token TK_while "while" 
%token TK_public "public"
%token TK_private "private"
%token TK_local "local"
%token TK_network "network"
%token TK_entities "entities"
%token TK_structure "structure" 
%token TK_external "external"

// Delimiters, separators, operators 

%token TK_colon
%token TK_colon_equals
%token TK_comma
%token TK_dash_dash_gt
%token TK_dash_gt
%token TK_dot
%token TK_dot_dot
%token TK_equals
%token TK_eqauls_equals_gt
%token TK_hash
%token TK_lbrace
%token TK_lbrack
%token TK_lpar
%token TK_lt
%token TK_gt
%token TK_plus
%token TK_qmark
%token TK_rbrace
%token TK_rbrack
%token TK_rpar
%token TK_semi
%token TK_star
%token TK_under_score
%token TK_vbar
%token TK_cinnamon_bun

%start compilation_unit

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

int main()
{
    return yyparse();
}