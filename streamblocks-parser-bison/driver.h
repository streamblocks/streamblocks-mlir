//
// Created by endrix on 27.05.21.
//

#ifndef DRIVER_H
#define DRIVER_H
# include <string>
# include <map>
# include "cal_parser.hpp"

// Give Flex the prototype of yylex we want ...
# define YY_DECL \
  cal::CalParser::symbol_type yylex (driver& drv)
// ... and declare it for the parser's sake.
YY_DECL;

// Conducting the whole scanning and parsing of Calc++.
class driver
{
public:
  driver () : trace_parsing (false), trace_scanning (false)
  {
    variables["one"] = 1;
    variables["two"] = 2;
  }

  std::map<std::string, int> variables;

  int result;

  // Run the parser on file F.  Return 0 on success.
  int parse (const std::string& f){
    file = f;
    location.initialize (&file);
    scan_begin ();
    cal::CalParser parse (*this);
    parse.set_debug_level (trace_parsing);
    int res = parse ();
    scan_end ();
    return res;
  }
  // The name of the file being parsed.
  std::string file;
  // Whether to generate parser debug traces.
  bool trace_parsing;

  // Handling the scanner.
  void scan_begin ();
  void scan_end ();
  // Whether to generate scanner debug traces.
  bool trace_scanning;
  // The token's location used by the scanner.
  cal::location location;
};

#endif // DRIVER_H
