#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>

#include "CalParser.h"
#include "CalParserConstants.h"
#include "CalParserTokenManager.h"
#include "CharStream.h"
#include "StreamBlocks/AST/AST.h"

using namespace cal::parser;
using namespace std;

JAVACC_STRING_TYPE ReadFileFully(char *file_name) {
  JAVACC_STRING_TYPE s;
#if WIDE_CHAR
  wifstream fp_in;
#else
  ifstream fp_in;
#endif
  fp_in.open(file_name, ios::in);
  // Very inefficient.
  while (!fp_in.eof()) {
    s += fp_in.get();
  }
  return s;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Usage: streamblocks-parser <cal input file>" << endl;
    exit(1);
  }
  JAVACC_STRING_TYPE s = ReadFileFully(argv[1]);
  CharStream *stream = new CharStream(s.c_str(), s.size() - 1, 1, 1);
  CalParserTokenManager *scanner = new CalParserTokenManager(stream);
  CalParser parser(scanner);
  parser.setErrorHandler(new ErrorHandler());
  parser.CompilationUnit();
  std::unique_ptr<cal::NamespaceDecl>
      root; // = (SimpleNode*)parser.jjtree.peekNode();
}
