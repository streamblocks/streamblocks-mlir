//
// Created by endrix on 19.05.21.
//

#ifndef QID_H
#define QID_H

#include <string>
#include <vector>

namespace cal {
class QID {
public:
  QID() {}

private:
  std::vector<std::string> parts;
};

} // namespace cal

#endif // QID_H
