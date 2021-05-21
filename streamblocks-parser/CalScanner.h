//
// Created by endrix on 21.05.21.
//

#ifndef CALSCANNER_H
#define CALSCANNER_H

#if !defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif

#include "cal_parser.hpp"
#include "location.hh"

namespace cal {

class CalScanner : public yyFlexLexer {
public:
  CalScanner(std::istream *in) : yyFlexLexer(in){};
  virtual ~CalScanner(){};

  // get rid of override virtual function warning
  using FlexLexer::yylex;

  virtual int yylex(cal::CalParser::semantic_type *const lval,
                    cal::CalParser::location_type *location);
  // YY_DECL defined in mc_lexer.l
  // Method body created by flex in mc_lexer.yy.cc

private:
  /* yyval ptr */
  cal::CalParser::semantic_type *yylval = nullptr;
};

} // namespace Cal

#endif // CALSCANNER_H
