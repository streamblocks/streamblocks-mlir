//
// Created by endrix on 21.05.21.
//

#ifndef CALDRIVER_H
#define CALDRIVER_H

#include <cstddef>
#include <fstream>
#include <istream>
#include <string>

#include "CalScanner.h"
#include "cal_parser.hpp"

#include "CalContext.h"

namespace cal {

class CalDriver {
public:
  CalDriver(CalContext &context)
      : context(context),  trace_scanning(false), trace_parsing(false) {}

  virtual ~CalDriver() {
    delete (scanner);
    scanner = nullptr;
    delete (parser);
    parser = nullptr;
  }

  /**
   * parse - parse from a file
   * @param filename - valid string with input file
   */
  void parse(const char *const filename) {
    assert(filename != nullptr);
    std::ifstream in_file(filename);
    if (!in_file.good()) {
      exit(EXIT_FAILURE);
    }
    parse_helper(in_file);
    return;
  }
  /**
   * parse - parse from a c++ input stream
   * @param is - std::istream&, valid input stream
   */
  void parse(std::istream &stream) {
    if (!stream.good() && stream.eof()) {
      return;
    }
    // else
    parse_helper(stream);
    return;
  }

  /// stream name (file or input stream) used for error messages.
  std::string streamname;

private:
  CalContext &context;

  CalParser *parser = nullptr;
  CalScanner *scanner = nullptr;

  /// enable debug output in the flex scanner
  bool trace_scanning;

  /// enable debug output in the bison parser
  bool trace_parsing;


  void parse_helper(std::istream &stream) {
    delete (scanner);
    // try {
    scanner = new cal::CalScanner(&stream);
    /*
    } catch (std::bad_alloc &ba) {
        std::cerr << "Failed to allocate scanner: (" << ba.what()
                  << "), exiting!!\n";
        exit(EXIT_FAILURE);
      }
  */
    delete (parser);
    // try {
    parser = new cal::CalParser((*scanner) /* scanner */, (*this) /* driver */);
    /*
    } catch (std::bad_alloc &ba) {
        std::cerr << "Failed to allocate parser: (" << ba.what()
                  << "), exiting!!\n";
        exit(EXIT_FAILURE);
      }*/

    const int accept(0);
    if (parser->parse() != accept) {
      std::cerr << "Parse failed!!\n";
    }
    return;
  }

};

} // namespace cal

#endif // CALDRIVER_H
