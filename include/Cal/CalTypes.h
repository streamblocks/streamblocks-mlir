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

  /// Signedness semantics.
  enum SignednessSemantics : uint32_t {
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }

  static IntType get(mlir::MLIRContext *ctx, int width, SignednessSemantics signedness = Signed);

  /// Return the bit width of this type.
  int getWidth();

  SignednessSemantics getSignedness() const;
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
