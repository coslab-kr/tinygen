#include "TensorEmitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

using namespace mlir;

Value tinygen::globalArrayAddress(OpBuilder &builder, Location location,
                                  llvm::StringRef name, Type element,
                                  llvm::ArrayRef<int64_t> shape, Value zero) {
  auto global = builder.create<emitc::GetGlobalOp>(
      location, emitc::ArrayType::get(shape, element),
      builder.getStringAttr(name));
  auto first = builder.create<emitc::SubscriptOp>(
      location, emitc::LValueType::get(element), global, ValueRange{zero});
  return builder.create<emitc::ApplyOp>(location,
                                        emitc::PointerType::get(element),
                                        builder.getStringAttr("&"), first);
}

void tinygen::initializeTensor(OpBuilder &builder, Location location,
                               llvm::StringRef name, RankedTensorType type,
                               Value data, Value zero) {
  auto tensor = builder.create<emitc::GetGlobalOp>(
      location,
      emitc::LValueType::get(
          emitc::OpaqueType::get(builder.getContext(), "Tensor")),
      builder.getStringAttr(name));
  auto assign = [&](llvm::StringRef field, Type fieldType, Value value) {
    auto member = builder.create<emitc::MemberOp>(
        location, emitc::LValueType::get(fieldType),
        builder.getStringAttr(field), tensor);
    builder.create<emitc::AssignOp>(location, member, value);
  };
  auto shape = globalArrayAddress(builder, location, name.str() + "_shape",
                                  builder.getI32Type(), {type.getRank()}, zero);
  assign("shape", emitc::PointerType::get(builder.getI32Type()), shape);
  auto rankGlobal = builder.create<emitc::GetGlobalOp>(
      location, emitc::LValueType::get(builder.getI32Type()),
      builder.getStringAttr(name.str() + "_rank"));
  auto rank =
      builder.create<emitc::LoadOp>(location, builder.getI32Type(), rankGlobal);
  assign("rank", builder.getI32Type(), rank);
  assign("location", emitc::PointerType::get(builder.getI8Type()), data);
}
