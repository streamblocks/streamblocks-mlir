//===- CalOps.cpp - Cal dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cal/CalOps.h"
#include "Cal/CalDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

using namespace streamblocks;
using namespace streamblocks::cal;

//===----------------------------------------------------------------------===//
// NamespaceOp
static LogicalResult verifyNamespaceOp(NamespaceOp op) {
  // -- TODO : Implement
  return success();
}

void NamespaceOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name) {
  using namespace mlir::function_like_impl;

  // Namespace QID
  result.addAttribute("qid", name);

  // Create a single-blocked region.
  result.addRegion();
  Region *regionBody = result.regions[0].get();
  Block *block = new Block();
  regionBody->push_back(block);
}

//===----------------------------------------------------------------------===//
// NetworkOp

static ParseResult parseNetworkOp(OpAsmParser &parser, OperationState &result) {
  return failure();
}

static void print(OpAsmPrinter &p, NetworkOp op) {}

static LogicalResult verifyNetworkOp(NetworkOp op) {
  // -- TODO : Implement
  return success();
}

//===----------------------------------------------------------------------===//
// Ports

/// Returns the type of the given component as a function type.
static FunctionType getComponentType(ActorOp actor) {
  return actor.getTypeAttr().getValue().cast<FunctionType>();
}

/// Returns the component port names in the given direction.
static ArrayAttr getComponentPortNames(ActorOp actor, PortDirection direction) {

  if (direction == PortDirection::INPUT)
    return actor.inPortNames();
  return actor.outPortNames();
}

/// Returns the port information for the given component.
SmallVector<PortInfo> cal::getPortInfo(Operation *op) {
  assert(isa<ActorOp>(op) && "Can only get port information from a component.");
  auto component = dyn_cast<ActorOp>(op);

  auto functionType = getComponentType(component);
  auto inPortTypes = functionType.getInputs();
  auto outPortTypes = functionType.getResults();
  auto inPortNamesAttr = getComponentPortNames(component, PortDirection::INPUT);
  auto outPortNamesAttr =
      getComponentPortNames(component, PortDirection::OUTPUT);

  SmallVector<PortInfo> results;
  for (size_t i = 0, e = inPortTypes.size(); i != e; ++i) {
    results.push_back({inPortNamesAttr[i].cast<StringAttr>(), inPortTypes[i],
                       PortDirection::INPUT});
  }
  for (size_t i = 0, e = outPortTypes.size(); i != e; ++i) {
    results.push_back({outPortNamesAttr[i].cast<StringAttr>(), outPortTypes[i],
                       PortDirection::OUTPUT});
  }
  return results;
}

/// Prints the port definitions of a Calyx component signature.
static void printPortDefList(OpAsmPrinter &p, ArrayRef<Type> portDefTypes,
                             ArrayAttr portDefNames) {
  p << '(';
  llvm::interleaveComma(
      llvm::zip(portDefNames, portDefTypes), p, [&](auto nameAndType) {
        if (auto name =
                std::get<0>(nameAndType).template dyn_cast<StringAttr>()) {
          p << '%' << name.getValue() << ": ";
        }
        p << std::get<1>(nameAndType);
      });
  p << ')';
}

//===----------------------------------------------------------------------===//
// ActorOp

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, MLIRContext *context,
                 OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &ports,
                 SmallVectorImpl<Type> &portTypes, StringRef attrName) {
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::OperandType port;
    Type portType;
    if (failed(parser.parseOptionalRegionArgument(port)) ||
        failed(parser.parseOptionalColon()) ||
        failed(parser.parseType(portType)))
      continue;
    ports.push_back(port);
    portTypes.push_back(portType);
  } while (succeeded(parser.parseOptionalComma()));

  // Add attribute for port names; these are currently
  // just inferred from the arguments of the component.
  SmallVector<Attribute> portNames(ports.size());
  llvm::transform(ports, portNames.begin(), [&](auto port) -> StringAttr {
    StringRef name = port.name;
    if (name.startswith("%"))
      name = name.drop_front();
    return StringAttr::get(context, name);
  });
  result.addAttribute(attrName, ArrayAttr::get(context, portNames));

  return (parser.parseRParen());
}

static ParseResult
parseIOSignature(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::OperandType> &inPorts,
                 SmallVectorImpl<Type> &inPortTypes,
                 SmallVectorImpl<OpAsmParser::OperandType> &outPorts,
                 SmallVectorImpl<Type> &outPortTypes) {
  auto *context = parser.getBuilder().getContext();
  if (parsePortDefList(parser, context, result, inPorts, inPortTypes,
                       "inPortNames") ||
      parser.parseArrow() ||
      parsePortDefList(parser, context, result, outPorts, outPortTypes,
                       "outPortNames"))
    return failure();

  return success();
}

static ParseResult parseActorOp(OpAsmParser &parser, OperationState &result) {

  using namespace mlir::function_like_impl;

  StringAttr componentName;
  if (parser.parseSymbolName(componentName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType> inPorts, outPorts;
  SmallVector<Type> inPortTypes, outPortTypes;
  if (parseIOSignature(parser, result, inPorts, inPortTypes, outPorts,
                       outPortTypes))
    return failure();

  // Build the component's type for FunctionLike trait.
  auto &builder = parser.getBuilder();
  auto type = builder.getFunctionType(inPortTypes, outPortTypes);
  result.addAttribute(ActorOp::getTypeAttrName(), TypeAttr::get(type));

  // The entry block needs to have same number of
  // input port definitions as the component.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, inPorts, inPortTypes))
    return failure();

  if (body->empty())
    body->push_back(new Block());

  return success();
}

static void print(OpAsmPrinter &p, ActorOp op) {
  auto componentName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << "cal.actor ";
  p.printSymbolName(componentName);

  auto functionType = getComponentType(op);
  auto inputPortTypes = functionType.getInputs();
  auto inputPortNames = op->getAttrOfType<ArrayAttr>("inPortNames");
  printPortDefList(p, inputPortTypes, inputPortNames);
  p << " -> ";
  auto outputPortTypes = functionType.getResults();
  auto outputPortNames = op->getAttrOfType<ArrayAttr>("outPortNames");
  printPortDefList(p, outputPortTypes, outputPortNames);

  p.printRegion(op.body(), /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);
}

static LogicalResult verifyActorOp(ActorOp op) {
  // -- TODO : Implement
  return success();
}

void ActorOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                    ArrayRef<PortInfo> ports) {
  using namespace mlir::function_like_impl;

  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> inPortTypes, outPortTypes;
  SmallVector<Attribute, 4> inPortNames, outPortNames;

  for (auto &&port : ports) {
    if (port.direction == PortDirection::INPUT) {
      inPortTypes.push_back(port.type);
      inPortNames.push_back(port.name);
    } else {
      outPortTypes.push_back(port.type);
      outPortNames.push_back(port.name);
    }
  }

  // Build the function type of the component.
  auto functionType = builder.getFunctionType(inPortTypes, outPortTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(functionType));

  // Record the port names of the component.
  result.addAttribute("inPortNames", builder.getArrayAttr(inPortNames));
  result.addAttribute("outPortNames", builder.getArrayAttr(outPortNames));

  // Create a single-blocked region.
  result.addRegion();
  Region *regionBody = result.regions[0].get();
  Block *block = new Block();
  regionBody->push_back(block);

  // Add input ports to the body block.
  for (auto port : ports) {
    if (port.direction == PortDirection::OUTPUT)
      continue;
    block->addArgument(port.type);
  }
}

//===----------------------------------------------------------------------===//
// ProcessOp

static ParseResult parseProcessOp(OpAsmParser &parser, OperationState &result) {
  return success();
}

static void print(OpAsmPrinter &p, ProcessOp op) {}

static LogicalResult verifyProcessOp(ProcessOp op) {
  // -- TODO : Implement
  return success();
}

void ProcessOp::build(mlir::OpBuilder &builder, mlir::OperationState &result, BoolAttr repeat){

  using namespace mlir::function_like_impl;

  // Is it a repeat process ?
  result.addAttribute("repeat", repeat);

  // Create a single-blocked region.
  result.addRegion();
  Region *regionBody = result.regions[0].get();
  Block *block = new Block();
  regionBody->push_back(block);
}

//===----------------------------------------------------------------------===//
// ConstantOp
static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference or array, then we expect a trailing
  // type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr, ArrayAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

static void print(OpAsmPrinter &printer, ConstantOp op) {
  printer << "cal.constant ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

static LogicalResult verifyConstantOp(ConstantOp op) {
  // -- TODO : Implement
  return success();
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       long value) {
  Type type = builder.getI32Type();
  ConstantOp::build(builder, state, type, builder.getIntegerAttr(type, value));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Cal/Cal.cpp.inc"
