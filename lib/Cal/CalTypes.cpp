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

#include "Cal/CalDialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace streamblocks {
namespace cal {
namespace detail {

/// This class represents the internal storage of the SCL `IntegerType` and
/// `LogicalType`.
struct BitWidthStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = int;

  /// A constructor for the type storage instance.
  BitWidthStorage(KeyTy width) : width(width) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == width; }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static BitWidthStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {

    // Copy the the provided `KeyTy` into the allocator.
    return new (allocator.allocate<BitWidthStorage>()) BitWidthStorage(key);
  }

  /// The bit-width of the type.
  KeyTy width;
};
} // end namespace detail

IntType IntType::get(mlir::MLIRContext *ctx, int width) {
  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in.
  // The other parameters are forwarded to the storage instance.
  return Base::get(ctx, width);
}

int IntType::getWidth() { return getImpl()->width; }

bool IntType::isSigned() { return false; }

} // namespace cal
} // namespace streamblocks

mlir::Type streamblocks::cal::CalDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if(keyword =="int"){

  }
}

void streamblocks::cal::CalDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {

}