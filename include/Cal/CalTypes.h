//===- CalStructure.td - Cal dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the types for Cal
//
//===----------------------------------------------------------------------===//

#ifndef CAL_DIALECT_CALTYPES_H
#define CAL_DIALECT_CALTYPES_H

#include "mlir/IR/Dialect.h"

namespace streamblocks {
namespace cal {
namespace detail {
struct BitWidthStorage;
} // end namespace detail

class IntType : public mlir::Type::TypeBase<IntType, mlir::Type,
                                            detail::BitWidthStorage> {
public:
  using Base::Base;

  static IntType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();

  /// Return if the integer is signed
  bool isSigned();
};

class LogicalType : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                                detail::BitWidthStorage> {
public:
  using Base::Base;

  static LogicalType get(mlir::MLIRContext *ctx, int width);

  /// Return the bit width of this type.
  int getWidth();
};

class RealType
    : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

} // namespace cal
} // namespace streamblocks

#endif // CAL_DIALECT_CALTYPES_H
