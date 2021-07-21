//
// Created by Endri Bezati on 20.07.21.
//

#ifndef CAL_DIALECT_CALOPS_H
#define CAL_DIALECT_CALOPS_H

#include "Cal/CalDialect.h"

namespace streamblocks {
namespace cal {

/// The direction of a CAL Port
enum PortDirection { INPUT = 0, OUTPUT = 1 };

/// This holds the name and type that describes the actor's or network's ports.
struct PortInfo {
  StringAttr name;
  Type type;
  PortDirection direction;
};

SmallVector<PortInfo> getPortInfo(Operation *op);

} // namespace cal
} // namespace streamblocks

#define GET_OP_CLASSES
#include "Cal/Cal.h.inc"

#endif // CAL_DIALECT_CALOPS_H
