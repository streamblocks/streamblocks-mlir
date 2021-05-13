// Identifier
%token ID


// Keywords
%token ACTION
%token ACTOR 
%token ALIAS 
%token ALL 
%token AND 
%token AS 
%token BEGIN 
%token CASE 
%token CONST 
%token DIV 
%token DO 
%token DOM 
%token ELSE 
%token ELSIF 
%token END 
%token ENDACTION 
%token ENDACTOR 
%token ENDCASE 
%token ENDCHOOSE 
%token ENDFOREACH 
%token ENDFUNCTION 
%token ENDIF 
%token ENDINITIALIZE 
%token ENDINVARIANT 
%token ENDLAMBDA 
%token ENDLET 
%token ENDPRIORITY 
%token ENDPROC 
%token ENDPROCEDURE 
%token ENDSCHEDULE 
%token ENDWHILE 
%token ENTITY 
%token FALSE 
%token FOR 
%token FOREACH 
%token FSM 
%token FUNCTION 
%token GUARD 
%token IF 
%token IMPORT 
%token IN 
%token INITIALIZE 
%token INVARIANT 
%token LAMBDA 
%token LET 
%token MAP 
%token MOD 
%token MULTI 
%token MUTABLE 
%token NAMESPACE 
%token NOT 
%token NULL 
%token OLD 
%token OF 
%token OR 
%token PRIORITY 
%token PROC 
%token PROCEDURE 
%token REGEXP 
%token REPEAT 
%token RNG 
%token SCHEDULE 
%token THEN 
%token TRUE 
%token TYPE 
%token VAR 
%token WHILE 
%token PUBLIC 
%token PRIVATE 
%token LOCAL 
%token NETWORK 
%token ENTITIES 
%token STRUCTURE 
%token EXTERNAL 

// Delimiters, separators, operators 

%token COLON
%token COLON_EQUALS
%token COMMA
%token DASH_DASH_GT
%token DASH_GT
%token DOT
%token DOTDOT
%token EQUALS
%token EQUALS_EQUALS_GT
%token HASH
%token LBRACE
%token LBRACK
%token LPAR
%token LT
%token GT
%token PLUS
%token QMARK
%token RBRACE
%token RBRACK
%token RPAR
%token SEMI
%token STAR
%token UNDER_SCORE
%token VBAR
%token CINNAMON_BUN

%%

CompilationUnit: NamespaceDeclContents
               | NamespaceDec
               ;

Annotation: CINNAMON_BUN ID
          | CINNAMON_BUN ID LPAR AnnotationParameters RPAR
          | CINNAMON_BUN ID LPAR RPAR
          ;

Annotations: /* empty */
           | Annotations Annotation
           ;

AnnotationParameters: AnnotationParameter
                    | AnnotationParameter COMMA AnnotationParameter
                    ;

AnnotationParameter: ID EQUALS Expression
                   | Expression
                   ;

NamespaceDec: NAMESPACE QID COLON NamespaceDeclContents END
            ;

NamespaceDeclContents: NamespaceDeclContent
                     | NamespaceDeclContent NamespaceDeclContent
                     ;

NamespaceDeclContent: Import SEMI
                    | ActorDecl
                    | NetworkDecl
                    | GlobalVarDecl
                    | GlobalTypeDecl
                    ;

Name: SimpleName
    | QID
    ; 

SimpleName: ID
          ;

QID: Name DOT ID
   ;

Availability: PUBLIC | PRIVATE | LOCAL 
            ;

VariableVarDecl: Type ID
               | Type ID EQUALS Expression
               | Type ID COLON_EQUALS Expression
               ;

FunctionalVarDecl: FUNCTION ID LPAR FormalValuePars RPAR END
                 | FUNCTION ID LPAR FormalValuePars RPAR FunctionBody END
                 | FUNCTION ID LPAR FormalValuePars RPAR DASH_DASH_GT Type FunctionBody END
                 ;


FunctionBody: VarDeclBlock SEMI Expression
            | SEMI Expression
            ;

ProcedureVarDecl: PROCEDURE ID LPAR FormalValuePars RPAR END
                | PROCEDURE ID LPAR FormalValuePars RPAR ProcedureBody END
                ;

ProcedureBody: VarDeclBlock 
             | VarDeclBlock BEGIN Statements
             | VarDeclBlock DO Statements
             ;


VarDeclBlock: VAR BlockVarDecls
            ;

BlockVarDecl: Annotations VariableVarDecl
            | Annotations FunctionalVarDecl
            | Annotations ProcedureVarDecl
            ;

BlockVarDecls: BlockVarDecl
             | BlockVarDecl COMMA BlockVarDecl
             ;

LocalVarDecl: 




%%