#include "Lowering.h"
#include <optional>
#include <set>

using namespace mlir;
using namespace tinygen;

namespace {
Operation *singleUser(Value value) {
  return value.hasOneUse() ? *value.user_begin() : nullptr;
}
} // namespace

SmallVector<Value, 4>
TosaToEmitCLowering::operandsToGlobalPtrs(Operation *op,
                                          PatternRewriter &rewriter) {

  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  SmallVector<Value, 4> operands;

  for (Value operand : op->getOperands()) {
    auto it = globalMap.find(operand.getAsOpaquePointer());
    if (it == globalMap.end()) {
      op->emitError() << "Operand has no matching global mapping.\n";
      continue;
    }

    std::string symbolName = it->second.getSymName().str();

    if (globalAddrMap.count(symbolName)) {
      operands.push_back(globalAddrMap[symbolName]);
      continue;
    }
    auto tensorTy = emitc::OpaqueType::get(ctx, "Tensor");
    auto ptrTy = emitc::PointerType::get(tensorTy);
    auto lvalueTy = emitc::LValueType::get(tensorTy);

    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        loc, lvalueTy, FlatSymbolRefAttr::get(ctx, symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(
        loc, ptrTy, rewriter.getStringAttr("&"), getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  return operands;
}

// const
LogicalResult
TosaToEmitCLowering::convertConstToEmitC(tosa::ConstOp constOp,
                                         PatternRewriter &rewriter) {
  auto *context = constOp.getContext();
  auto result = constOp.getResult();
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(constOp.getResult().getType());
  if (!resultType) {
    return failure();
  }

  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    constOp.emitError("Global tensor not found in map for this SSA value");
    return failure();
  }

  std::string varName = it->second.getSymName().str();
  rewriter.eraseOp(constOp);
  return success();
}

// conv2d
LogicalResult
TosaToEmitCLowering::convertConv2DToEmitC(tosa::Conv2DOp convOp,
                                          PatternRewriter &rewriter) {

  if (!convOp->hasOneUse())
    return failure();
  auto rescaleOp =
      dyn_cast<tosa::RescaleOp>(*convOp->getResult(0).user_begin());
  if (!rescaleOp)
    return failure();

  tosa::ClampOp clampOp = nullptr;
  if (rescaleOp->hasOneUse()) {
    clampOp = dyn_cast<tosa::ClampOp>(*rescaleOp->getResult(0).user_begin());
  }

  Value filter = convOp.getWeight();
  auto filterType = mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());
  if (!filterType || !filterType.hasStaticShape())
    return failure();

  StringAttr callee;

  auto shape = filterType.getShape();
  int64_t filter_size = shape[2];
  int64_t filter_size2 = shape[1];
  auto stride = convOp.getStride();

  if (filter_size == filter_size2) {
    if (filter_size == 1) {
      if ((stride[0] == 1) && (stride[1] == 1)) {
        callee = rewriter.getStringAttr("conv2d_f1x1_s1x1_s8");
      } else {
        callee = rewriter.getStringAttr("conv2d_f1x1_s8");
      }
    } else if (filter_size == 3) {
      callee = rewriter.getStringAttr("conv2d_f3x3_s8");
    } else {
      callee = rewriter.getStringAttr("conv2d_fnxn_s8");
    }
  } else {
    callee = rewriter.getStringAttr("conv2d_fnxm_s8");
  }

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  llvm::SmallVector<Attribute, 8> argAttrs = {
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(2),
      rewriter.getIndexAttr(3),
      rewriter.getIndexAttr(4),
      rewriter.getIndexAttr(5),
      rewriter.getIndexAttr(6),
      rewriter.getI64TensorAttr(convOp.getPad()),
      rewriter.getI64TensorAttr(convOp.getStride()),
      rewriter.getI64TensorAttr(convOp.getDilation()),
  };

  auto quantInfo = convOp.getQuantizationInfo();

  if (quantInfo.has_value()) {
    argAttrs.push_back(rewriter.getI32IntegerAttr(quantInfo->getInputZp()));
  } else {
    return rewriter.notifyMatchFailure(
        convOp, "int8 convolution requires quantization_info");
  }

  argAttrs.push_back(rescaleOp.getOutputZpAttr());

  if (clampOp) {
    argAttrs.push_back(clampOp.getMinIntAttr());
    argAttrs.push_back(clampOp.getMaxIntAttr());
  } else {
    argAttrs.push_back(rewriter.getI32IntegerAttr(-128));
    argAttrs.push_back(rewriter.getI32IntegerAttr(127));
  }

  ArrayAttr args = rewriter.getArrayAttr(argAttrs);

  ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(lvalueType)});
  auto operands = operandsToGlobalPtrs(convOp.getOperation(), rewriter);

  auto ctx = rewriter.getContext();
  auto i32Type = rewriter.getI32Type();

  if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
    zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
  }
  Value rescaleResult = rescaleOp.getResult();
  auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
  if (itRescale == globalMap.end())
    return convOp.emitError("Missing global mapping for rescale result");
  std::string rescaleSymbol = itRescale->second.getSymName().str();

  Value convResult = convOp.getResult();
  auto itConv = globalMap.find(convResult.getAsOpaquePointer());
  if (itConv == globalMap.end())
    return convOp.emitError("Missing global mapping for conv2d result");
  std::string convSymbol = itConv->second.getSymName().str();

  // multiplier
  int64_t multiplierSize = rescaleOp.getMultiplierAttr().size();

  auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
  auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), multiplierArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier"));
  auto arr0M = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), multiplierGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0M);
  operands.push_back(multiplierPtr.getResult());

  // shift
  int64_t shiftSize = rescaleOp.getShiftAttr().size();
  auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
  auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), shiftArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift"));
  auto arr0S = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), shiftGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto shiftPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0S);
  operands.push_back(shiftPtr.getResult());

  if (globalAddrMap.count(rescaleSymbol)) {
    operands.push_back(globalAddrMap[rescaleSymbol]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        rescaleOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(ctx, rescaleSymbol));
    auto addrOf = rewriter.create<emitc::ApplyOp>(rescaleOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[rescaleSymbol] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  Value finalResult = clampOp ? clampOp.getResult() : rescaleOp.getResult();
  auto it = globalMap.find(finalResult.getAsOpaquePointer());
  if (it == globalMap.end())
    return convOp.emitError("Missing global mapping for fused conv result");
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        convOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(convOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  auto padAttr = convOp.getPad();
  auto strideAttr = convOp.getStride();
  auto dilationAttr = convOp.getDilation();

  std::vector<int64_t> params = {padAttr[0],      padAttr[2],
                                 strideAttr[0],   strideAttr[1],
                                 dilationAttr[0], dilationAttr[1]};
  if (callee.str() == "conv2d_f1x1_s1x1_s8") {
    params.clear();
  }

  auto key = callee.str();

  if (callee.str() == "conv2d_fnxm_s8" || callee.str() == "conv2d_fnxn_s8") {
    key = "conv2d_f3x3_s8";
    convParamMap[callee.str()] = {};
  }
  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      convOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);

  auto it2 = convParamMap.find(key);
  if (it2 == convParamMap.end()) {
    convParamMap[key] = params;
  } else {
    auto &oldParams = it2->second;
    if (oldParams.size() == params.size()) {
      for (size_t i = 0; i < oldParams.size(); ++i) {
        if (oldParams[i] != params[i]) {
          oldParams[i] = -1;
        }
      }
    }
  }

  if (clampOp)
    rewriter.eraseOp(clampOp);
  rewriter.eraseOp(rescaleOp);
  rewriter.eraseOp(convOp);
  return success();
}

// depthwise_conv2d
LogicalResult TosaToEmitCLowering::convertDepthwiseConv2DToEmitC(
    tosa::DepthwiseConv2DOp convOp, PatternRewriter &rewriter) {

  if (!convOp->hasOneUse())
    return failure();
  auto rescaleOp =
      dyn_cast<tosa::RescaleOp>(*convOp->getResult(0).user_begin());
  if (!rescaleOp)
    return failure();

  Value filter = convOp.getWeight();
  auto filterType = mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());
  if (!filterType || !filterType.hasStaticShape())
    return failure();

  StringAttr callee;
  auto shape = filterType.getShape();
  int64_t filter_size = shape[0];
  int64_t filter_size2 = shape[1];

  if (filter_size == filter_size2) {
    if (filter_size == 1) {
      callee = rewriter.getStringAttr("depthwise_conv2d_f1x1_s8");
    } else if (filter_size == 3) {
      callee = rewriter.getStringAttr("depthwise_conv2d_f3x3_s8");
    } else {
      callee = rewriter.getStringAttr("depthwise_conv2d_fnxn_s8");
    }
  } else {
    callee = rewriter.getStringAttr("depthwise_conv2d_fnxm_s8");
  }

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  llvm::SmallVector<Attribute, 8> argAttrs = {
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(2),
      rewriter.getIndexAttr(3),
      rewriter.getIndexAttr(4),
      rewriter.getIndexAttr(5),
      rewriter.getI64TensorAttr(convOp.getPad()),
      rewriter.getI64TensorAttr(convOp.getStride()),
      rewriter.getI64TensorAttr(convOp.getDilation()),
  };

  auto quantInfo = convOp.getQuantizationInfo();
  if (quantInfo.has_value()) {
    argAttrs.push_back(rewriter.getI32IntegerAttr(quantInfo->getInputZp()));
  } else {
    return rewriter.notifyMatchFailure(
        convOp, "int8 convolution requires quantization_info");
  }
  argAttrs.push_back(rescaleOp.getOutputZpAttr());

  ArrayAttr args = rewriter.getArrayAttr(argAttrs);
  ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(lvalueType)});
  auto operands = operandsToGlobalPtrs(convOp.getOperation(), rewriter);

  auto ctx = rewriter.getContext();
  auto i32Type = rewriter.getI32Type();

  if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
    zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
  }

  Value rescaleResult = rescaleOp.getResult();
  auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
  if (itRescale == globalMap.end())
    return convOp.emitError("Missing global mapping for rescale result");
  std::string rescaleSymbol = itRescale->second.getSymName().str();

  Value convResult = convOp.getResult();
  auto itConv = globalMap.find(convResult.getAsOpaquePointer());
  if (itConv == globalMap.end())
    return convOp.emitError(
        "Missing global mapping for depthwise conv2d result");
  std::string convSymbol = itConv->second.getSymName().str();

  // multiplier
  int64_t multiplierSize = rescaleOp.getMultiplierAttr().size();
  auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
  auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), multiplierArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier"));
  auto arr0M = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), multiplierGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0M);
  operands.push_back(multiplierPtr.getResult());

  // shift
  int64_t shiftSize = rescaleOp.getShiftAttr().size();
  auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
  auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), shiftArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift"));
  auto arr0S = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), shiftGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto shiftPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0S);
  operands.push_back(shiftPtr.getResult());

  if (globalAddrMap.count(rescaleSymbol)) {
    operands.push_back(globalAddrMap[rescaleSymbol]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        rescaleOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(ctx, rescaleSymbol));
    auto addrOf = rewriter.create<emitc::ApplyOp>(rescaleOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[rescaleSymbol] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  Value finalResult = rescaleOp.getResult();
  auto it = globalMap.find(finalResult.getAsOpaquePointer());
  if (it == globalMap.end())
    return convOp.emitError(
        "Missing global mapping for fused depthwise conv result");
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        convOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(convOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  auto padAttr = convOp.getPad();
  auto strideAttr = convOp.getStride();
  auto dilationAttr = convOp.getDilation();

  std::vector<int64_t> params = {padAttr[0],      padAttr[2],
                                 strideAttr[0],   strideAttr[1],
                                 dilationAttr[0], dilationAttr[1]};
  if (callee.str() == "depthwise_conv2d_f1x1_s8") {
    params.clear();
  }
  auto key = callee.str();

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      convOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);

  auto it2 = convParamMap.find(key);
  if (it2 == convParamMap.end()) {
    convParamMap[key] = params;
  } else {
    auto &oldParams = it2->second;
    if (oldParams.size() == params.size()) {
      for (size_t i = 0; i < oldParams.size(); ++i) {
        if (oldParams[i] != params[i]) {
          oldParams[i] = -1;
        }
      }
    }
  }

  rewriter.eraseOp(rescaleOp);
  rewriter.eraseOp(convOp);
  return success();
}

// maxpool
LogicalResult
TosaToEmitCLowering::convertMax_pool2DToEmitC(tosa::MaxPool2dOp max_poolOp,
                                              PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("max_pool_2d");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      rewriter.getI64TensorAttr(max_poolOp.getPad()),
      rewriter.getI64TensorAttr(max_poolOp.getStride()),
      rewriter.getI64TensorAttr(max_poolOp.getKernel()),
  });

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(max_poolOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(max_poolOp.getOperation(), rewriter);
  Value result = max_poolOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return max_poolOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
      max_poolOp.getLoc(), lvalueType,
      FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
  auto addrOf = rewriter.create<emitc::ApplyOp>(max_poolOp.getLoc(), ptrType,
                                                rewriter.getStringAttr("&"),
                                                getGlobal.getResult());
  operands.push_back(addrOf.getResult());

  auto padAttr = max_poolOp.getPad();
  auto strideAttr = max_poolOp.getStride();
  auto kernelAttr = max_poolOp.getKernel();
  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      max_poolOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);

  std::vector<int64_t> params = {padAttr[0],    padAttr[2],    strideAttr[0],
                                 strideAttr[1], kernelAttr[0], kernelAttr[1]};

  auto key = callee.str();
  auto it2 = convParamMap.find(key);

  if (it2 == convParamMap.end()) {
    convParamMap[key] = params;
  } else {
    auto &oldParams = it2->second;
    if (oldParams.size() == params.size()) {
      for (size_t i = 0; i < oldParams.size(); ++i) {
        if (oldParams[i] != params[i]) {
          oldParams[i] = -1;
        }
      }
    }
  }

  rewriter.eraseOp(max_poolOp);
  return success();
}

// avg_pool2d
LogicalResult
TosaToEmitCLowering::convertAvg_pool2DToEmitC(tosa::AvgPool2dOp avg_poolOp,
                                              PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("avg_pool_2d");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      rewriter.getI64TensorAttr(avg_poolOp.getPad()),
      rewriter.getI64TensorAttr(avg_poolOp.getStride()),
      rewriter.getI64TensorAttr(avg_poolOp.getKernel()),
  });

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(avg_poolOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(avg_poolOp.getOperation(), rewriter);

  Value result = avg_poolOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return avg_poolOp.emitError("Missing global mapping for avg_pool2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
      avg_poolOp.getLoc(), lvalueType,
      FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

  auto addrOf = rewriter.create<emitc::ApplyOp>(avg_poolOp.getLoc(), ptrType,
                                                rewriter.getStringAttr("&"),
                                                getGlobal.getResult());
  operands.push_back(addrOf.getResult());

  auto padAttr = avg_poolOp.getPad();
  auto strideAttr = avg_poolOp.getStride();
  auto kernelAttr = avg_poolOp.getKernel();

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      avg_poolOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);

  std::vector<int64_t> params = {padAttr[0],    padAttr[2],    strideAttr[0],
                                 strideAttr[1], kernelAttr[0], kernelAttr[1]};

  auto key = callee.str();
  auto it2 = convParamMap.find(key);

  if (it2 == convParamMap.end()) {
    convParamMap[key] = params;
  } else {
    auto &oldParams = it2->second;
    if (oldParams.size() == params.size()) {
      for (size_t i = 0; i < oldParams.size(); ++i) {
        if (oldParams[i] != params[i]) {
          oldParams[i] = -1;
        }
      }
    }
  }

  rewriter.eraseOp(avg_poolOp);
  return success();
}

// pad
LogicalResult
TosaToEmitCLowering::convertPadToEmitC(tosa::PadOp padOp,
                                       PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("pad_s8");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args = rewriter.getArrayAttr(
      {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
       rewriter.getIndexAttr(2), rewriter.getIndexAttr(3)});
  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(padOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(padOp.getOperation(), rewriter);

  Value result = padOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return padOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
      padOp.getLoc(), lvalueType,
      FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
  auto addrOf = rewriter.create<emitc::ApplyOp>(padOp.getLoc(), ptrType,
                                                rewriter.getStringAttr("&"),
                                                getGlobal.getResult());
  operands.push_back(addrOf.getResult());

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      padOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  convParamMap[callee.str()] = {};
  rewriter.eraseOp(padOp);
  return success();
}

// const_shape
LogicalResult
TosaToEmitCLowering::convertConstShapeToEmitC(tosa::ConstShapeOp constShapeOp,
                                              PatternRewriter &rewriter) {
  auto *context = constShapeOp.getContext();
  auto result = constShapeOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    constShapeOp.emitError("Global tensor not found in map for this SSA value");
    return failure();
  }
  rewriter.eraseOp(constShapeOp);
  return success();
}

// concat
LogicalResult
TosaToEmitCLowering::convertConcatToEmitC(tosa::ConcatOp concatOp,
                                          PatternRewriter &rewriter) {

  Location loc = concatOp.getLoc();
  IntegerAttr axisAttr = concatOp.getAxisAttr();
  int64_t axis = axisAttr.getInt();

  StringAttr callee;
  if (axis == 0) {
    callee = rewriter.getStringAttr("concat_axis0");
  } else if (axis == 1) {
    callee = rewriter.getStringAttr("concat_axis1");
  } else if (axis == 2) {
    callee = rewriter.getStringAttr("concat_axis2");
  } else {
    callee = rewriter.getStringAttr("concat_axis3");
  }

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(2), concatOp.getAxisAttr()});

  ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(lvalueType)});

  auto now_concatOp = concatOp;

  SmallVector<Value, 4> operands_concat;

  for (Value operand : concatOp.getOperands()) {
    auto it = globalMap.find(operand.getAsOpaquePointer());
    if (it == globalMap.end()) {
      concatOp.emitError() << "Missing global tensor for operand";
      return failure();
    }
    std::string symbolName = it->second.getSymName().str();

    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        loc, lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(
        loc, ptrType, rewriter.getStringAttr("&"), getGlobal.getResult());
    operands_concat.push_back(addrOf.getResult());
  }

  Value result = concatOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return concatOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands_concat.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        concatOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(concatOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());

    globalAddrMap[symbolName] = addrOf.getResult();
    operands_concat.push_back(addrOf.getResult());
  }

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, callee, args, ArrayAttr(), operands_concat);
  convParamMap[callee.str()] = {};
  rewriter.eraseOp(concatOp);
  return success();
}

// cast
LogicalResult
TosaToEmitCLowering::convertCastToEmitC(tosa::CastOp castOp,
                                        PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("cast");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  auto inputType = castOp.getInput().getType();
  auto outputType = castOp.getResult().getType();

  auto inTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
  auto outTensor = mlir::dyn_cast<mlir::RankedTensorType>(outputType);

  int quant_check = -1;
  if (inTensor && outTensor) {
    auto inElemType = inTensor.getElementType();
    auto outElemType = outTensor.getElementType();

    if (inElemType.isInteger(8) && outElemType.isF32()) {
      quant_check = 0;
    } else if (auto quantType =
                   mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
                       inElemType)) {
      if (quantType.getStorageTypeIntegralWidth() == 8 && outElemType.isF32()) {
        quant_check = 0;
      }
    } else if (inElemType.isF32() && outElemType.isInteger(8)) {
      quant_check = 1;
    } else if (inElemType.isF32()) {
      if (auto quantType =
              mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType)) {
        if (quantType.getStorageTypeIntegralWidth() == 8) {
          quant_check = 1;
        }
      }
    } else {
      quant_check = -1;
    }

  } else {
    quant_check = -1;
  }

  SmallVector<mlir::Attribute, 4> argsVec = {
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
      rewriter.getI32IntegerAttr(quant_check)};
  ArrayAttr args = rewriter.getArrayAttr(argsVec);

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(castOp.getResult().getType())});
  auto operands = operandsToGlobalPtrs(castOp.getOperation(), rewriter);

  Value result = castOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return castOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        castOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(castOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());

    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};
  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      castOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(castOp);
  return success();
}

// transpose_conv2d
LogicalResult TosaToEmitCLowering::convertTransposeConv2DToEmitC(
    tosa::TransposeConv2DOp transposeConvOp, PatternRewriter &rewriter) {

  if (!transposeConvOp->hasOneUse())
    return failure();
  auto rescaleOp =
      dyn_cast<tosa::RescaleOp>(*transposeConvOp->getResult(0).user_begin());
  if (!rescaleOp)
    return failure();

  auto strideAttr = transposeConvOp.getStride();

  auto inputType =
      mlir::cast<mlir::ShapedType>(transposeConvOp.getInput().getType());
  auto inputShape = inputType.getShape();
  int64_t input_ch = inputShape[3];

  StringAttr callee;

  if (strideAttr[0] <= 2 && strideAttr[1] <= 2 && input_ch > 16) {
    callee = rewriter.getStringAttr("transpose_conv2d_s1_s2_s8");
  } else {
    callee = rewriter.getStringAttr("transpose_conv2d_s8");
  }

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  llvm::SmallVector<Attribute, 8> argAttrs = {
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(2),
      rewriter.getIndexAttr(3),
      rewriter.getIndexAttr(4),
      rewriter.getIndexAttr(5),
      rewriter.getI64TensorAttr(transposeConvOp.getOutPad()),
      rewriter.getI64TensorAttr(transposeConvOp.getStride()),
      rewriter.getI64TensorAttr(transposeConvOp.getOutShape())};
  auto quantInfo = transposeConvOp.getQuantizationInfo();

  if (quantInfo.has_value()) {
    argAttrs.push_back(rewriter.getI32IntegerAttr(quantInfo->getInputZp()));
  } else {
    return rewriter.notifyMatchFailure(
        transposeConvOp, "int8 convolution requires quantization_info");
  }

  argAttrs.push_back(rescaleOp.getOutputZpAttr());

  ArrayAttr args = rewriter.getArrayAttr(argAttrs);

  auto operands =
      operandsToGlobalPtrs(transposeConvOp.getOperation(), rewriter);

  auto ctx = rewriter.getContext();
  auto i32Type = rewriter.getI32Type();

  Value transResult = transposeConvOp.getResult();
  auto itTrans = globalMap.find(transResult.getAsOpaquePointer());
  if (itTrans == globalMap.end())
    return transposeConvOp.emitError(
        "Missing global mapping for transpose result");
  std::string transSymbol = itTrans->second.getSymName().str();

  if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
    zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
  }
  Value rescaleResult = rescaleOp.getResult();
  auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
  if (itRescale == globalMap.end())
    return transposeConvOp.emitError(
        "Missing global mapping for rescale result");
  std::string rescaleSymbol = itRescale->second.getSymName().str();

  //  multiplier
  int64_t multiplierSize = rescaleOp.getMultiplierAttr().size();
  auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
  auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), multiplierArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier"));
  auto arr0M = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), multiplierGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0M);
  operands.push_back(multiplierPtr.getResult());

  // shift
  int64_t shiftSize = rescaleOp.getShiftAttr().size();
  auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
  auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
      rescaleOp.getLoc(), shiftArrayType,
      FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift"));
  auto arr0S = rewriter.create<emitc::SubscriptOp>(
      rescaleOp.getLoc(), emitc::LValueType::get(i32Type), shiftGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto shiftPtr = rewriter.create<emitc::ApplyOp>(
      rescaleOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0S);
  operands.push_back(shiftPtr.getResult());

  // output tensor pointer
  Value finalResult = rescaleOp.getResult();
  auto it = globalMap.find(finalResult.getAsOpaquePointer());
  if (it == globalMap.end())
    return transposeConvOp.emitError(
        "Missing global mapping for fused transpose result");
  std::string symbolName = it->second.getSymName().str();
  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        transposeConvOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(ctx, symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(
        transposeConvOp.getLoc(), ptrType, rewriter.getStringAttr("&"),
        getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }
  auto padAttr = transposeConvOp.getOutPad();
  auto filterType =
      mlir::cast<mlir::ShapedType>(transposeConvOp.getFilter().getType());
  auto filterShape = filterType.getShape();

  int64_t filter_h = filterShape[1];
  int64_t filter_w = filterShape[2];

  int64_t out_pad_h = padAttr[0];
  int64_t out_pad_w = padAttr[2];

  int64_t padding_h = filter_h - 1 - out_pad_h;
  int64_t padding_w = filter_w - 1 - out_pad_w;

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(transposeConvOp.getLoc(),
                                                      TypeRange{}, callee, args,
                                                      ArrayAttr(), operands);

  std::vector<int64_t> params = {padding_h,
                                 padding_w,
                                 rewriter.getI64IntegerAttr(1).getInt(),
                                 rewriter.getI64IntegerAttr(1).getInt(),
                                 rewriter.getI64IntegerAttr(1).getInt(),
                                 rewriter.getI64IntegerAttr(1).getInt()};
  auto key = callee.str();

  if (callee.str() == "transpose_conv2d_s1_s2_s8") {
    key = "conv2d_f3x3_s8";
  }

  auto it2 = convParamMap.find(key);
  if (it2 == convParamMap.end()) {
    convParamMap[key] = params;
  } else {
    auto &oldParams = it2->second;
    if (oldParams.size() == params.size()) {
      for (size_t i = 0; i < oldParams.size(); ++i) {
        if (oldParams[i] != params[i]) {
          oldParams[i] = -1;
        }
      }
    }
  }
  convParamMap[callee.str()] = {};
  rewriter.eraseOp(rescaleOp);
  rewriter.eraseOp(transposeConvOp);
  return success();
}

// transpose
LogicalResult
TosaToEmitCLowering::convertTransposeToEmitC(tosa::TransposeOp transposeOp,
                                             PatternRewriter &rewriter) {
  Value result = transposeOp.getResult();
  Type resultType = result.getType();
  StringAttr callee = rewriter.getStringAttr("transpose");

  llvm::errs() << "[DEBUG] Selected callee: " << callee.getValue() << "\n";

  if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
    zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
  }
  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  auto operands = operandsToGlobalPtrs(transposeOp.getOperation(), rewriter);

  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(2)});

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(transposeOp.getResult().getType())});

  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return transposeOp.emitError("Missing global mapping for transpose result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        transposeOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(transposeOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());

    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      transposeOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(transposeOp);
  return success();
}

// add
LogicalResult
TosaToEmitCLowering::convertAddToEmitC(tosa::AddOp addOp,
                                       PatternRewriter &rewriter) {
  if (!cast<ShapedType>(addOp.getType()).getElementType().isF32())
    return rewriter.notifyMatchFailure(addOp, "float add requires f32 tensors");
  Value result = addOp.getResult();
  Type resultType = addOp.getResult().getType();
  StringAttr callee = rewriter.getStringAttr("add_float");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  auto operands = operandsToGlobalPtrs(addOp.getOperation(), rewriter);

  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(2)});
  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(addOp.getResult().getType())});

  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return addOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        addOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(addOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());

    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      addOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(addOp);
  return success();
}

// add_c & Maximum
LogicalResult
TosaToEmitCLowering::convertRescaleToEmitC(tosa::RescaleOp rescaleOp1,
                                           PatternRewriter &rewriter) {
  auto addUser = singleUser(rescaleOp1.getResult());
  auto addOp = dyn_cast_or_null<tosa::AddOp>(addUser);
  tosa::RescaleOp rescaleOp2 = nullptr;
  tosa::RescaleOp rescaleOp3 = nullptr;
  tosa::RescaleOp rescaleOp4 = nullptr;
  tosa::RescaleOp rescale_lhs = nullptr;
  tosa::RescaleOp rescale_temp = nullptr;
  tosa::RescaleOp rescale_rhs = nullptr;
  int check = 0;
  tosa::MaximumOp maxOp = nullptr;
  maxOp = dyn_cast_or_null<tosa::MaximumOp>(addUser);

  if (!addOp && !maxOp) {
    Operation *user1 = singleUser(rescaleOp1.getResult());
    if (!user1)
      return failure();
    rescaleOp2 = dyn_cast<tosa::RescaleOp>(user1);
    if (!rescaleOp2) {
      return failure();
    }

    // Check whether the user of rescale2 is add
    addOp = dyn_cast_or_null<tosa::AddOp>(singleUser(rescaleOp2.getResult()));
    if (!addOp) {
      return failure();
    }

    // The user of add is rescale4
    auto rescaleUser = singleUser(addOp.getResult());
    rescaleOp4 = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
    if (!rescaleOp4) {
      return failure();
    }

    // One of the operands of add should be the rescale2 we just found
    rescaleOp3 = addOp.getOperand(1).getDefiningOp<tosa::RescaleOp>();
    if (!rescaleOp3) {
      return failure();
    }
    rescale_lhs = rescaleOp1;
    rescale_temp = rescaleOp2;
    rescale_rhs = rescaleOp3;
  } else if (addOp && !maxOp) {
    check = 1;

    auto rescaleUser = singleUser(addOp.getResult());
    rescaleOp4 = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
    if (!rescaleOp4) {
      return failure();
    }

    rescaleOp3 = addOp.getOperand(1).getDefiningOp<tosa::RescaleOp>();
    if (!rescaleOp3) {
      return failure();
    }

    rescaleOp2 = rescaleOp3.getOperand().getDefiningOp<tosa::RescaleOp>();
    if (!rescaleOp2) {
      return failure();
    }
    rescale_lhs = rescaleOp1;
    rescale_temp = rescaleOp3;
    rescale_rhs = rescaleOp2;

  } else if (maxOp) {
    rescaleOp2 = maxOp.getOperand(0).getDefiningOp<tosa::RescaleOp>();
    auto rescaleUser = singleUser(maxOp.getResult());
    rescaleOp3 = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
  }

  if (rescaleOp2 && rescaleOp3 && addOp && rescaleOp4 && !maxOp) {
    Value result = addOp.getResult();
    Type resultType = result.getType();
    StringAttr callee = rewriter.getStringAttr("add_s8");

    auto opaqueTensorType =
        emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
    auto lvalueType = emitc::LValueType::get(opaqueTensorType);
    auto ptrType = emitc::PointerType::get(opaqueTensorType);

    // --- ZP attributes (rescale1, rescale2, rescale3, rescale4) ---
    auto lhsInZp = rescale_lhs.getInputZpAttr();
    auto tempInZp = rescale_temp.getInputZpAttr();
    auto rhsInZp2 = rescale_rhs.getInputZpAttr();
    auto outOutZp = rescaleOp4.getOutputZpAttr();

    SmallVector<Attribute> argAttrs = {
        rewriter.getIndexAttr(0),  rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(2),  rewriter.getIndexAttr(3),
        rewriter.getIndexAttr(4),  rewriter.getIndexAttr(5),
        rewriter.getIndexAttr(6),  rewriter.getIndexAttr(7),
        rewriter.getIndexAttr(8),  rewriter.getIndexAttr(9),
        rewriter.getIndexAttr(10), rewriter.getIndexAttr(11),
        rewriter.getIndexAttr(12), rewriter.getIndexAttr(13),
        rewriter.getIndexAttr(14)};
    auto checkAttr = rewriter.getI32IntegerAttr(check);
    argAttrs.push_back(checkAttr);
    argAttrs.push_back(lhsInZp);
    argAttrs.push_back(tempInZp);
    argAttrs.push_back(rhsInZp2);
    argAttrs.push_back(outOutZp);
    ArrayAttr args = rewriter.getArrayAttr(argAttrs);

    auto ctx = rewriter.getContext();
    auto i32Type = rewriter.getIntegerType(32);

    if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
      zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                     rewriter.getIndexType(),
                                                     rewriter.getIndexAttr(0));
    }

    SmallVector<Value> operands;

    {
      Value rescaleResult = rescale_lhs.getOperand();
      auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
      if (itRescale == globalMap.end())
        return rescaleOp1.emitError(
            "Missing global mapping for rescale lhs result");

      std::string rescaleSymbol = itRescale->second.getSymName().str();

      if (globalAddrMap.count(rescaleSymbol)) {
        operands.push_back(globalAddrMap[rescaleSymbol]);
      } else {
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp1.getLoc(), lvalueType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol));

        auto addrOf = rewriter.create<emitc::ApplyOp>(
            rescaleOp1.getLoc(), ptrType, rewriter.getStringAttr("&"),
            getGlobal.getResult());

        globalAddrMap[rescaleSymbol] = addrOf.getResult();
        operands.push_back(addrOf.getResult());
      }
    }
    // input2: either rescaleOp3 or rescale_temp depending on 'check'
    {
      Value rescaleResult = rescale_rhs.getOperand();
      auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
      if (itRescale == globalMap.end())
        return rescaleOp1.emitError(
            "Missing global mapping for rhs rescale result");

      std::string rescaleSymbol = itRescale->second.getSymName().str();
      if (globalAddrMap.count(rescaleSymbol)) {
        operands.push_back(globalAddrMap[rescaleSymbol]);
      } else {
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp1.getLoc(), lvalueType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol));

        auto addrOf = rewriter.create<emitc::ApplyOp>(
            rescaleOp1.getLoc(), ptrType, rewriter.getStringAttr("&"),
            getGlobal.getResult());
        globalAddrMap[rescaleSymbol] = addrOf.getResult();
        operands.push_back(addrOf.getResult());
      }
    }
    // Helper:add multiplier/shift and the global pointer operand
    auto addMultShiftOperands = [&](tosa::RescaleOp r) -> LogicalResult {
      Value rescaleResult = r.getResult();
      auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
      if (itRescale == globalMap.end())
        return r.emitError("Missing global mapping for rescale result");

      std::string rescaleSymbol = itRescale->second.getSymName().str();

      // multiplier
      int64_t multiplierSize = r.getMultiplierAttr().size();
      auto multiplierArrayType =
          emitc::ArrayType::get({multiplierSize}, i32Type);
      auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
          r.getLoc(), multiplierArrayType,
          FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier"));
      auto arr0M = rewriter.create<emitc::SubscriptOp>(
          r.getLoc(), emitc::LValueType::get(i32Type), multiplierGlobal,
          SmallVector<Value, 1>{zeroIndex});
      auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
          r.getLoc(), emitc::PointerType::get(i32Type),
          rewriter.getStringAttr("&"), arr0M);
      operands.push_back(multiplierPtr.getResult());

      // shift
      int64_t shiftSize = r.getShiftAttr().size();
      auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
      auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
          r.getLoc(), shiftArrayType,
          FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift"));
      auto arr0S = rewriter.create<emitc::SubscriptOp>(
          r.getLoc(), emitc::LValueType::get(i32Type), shiftGlobal,
          SmallVector<Value, 1>{zeroIndex});
      auto shiftPtr = rewriter.create<emitc::ApplyOp>(
          r.getLoc(), emitc::PointerType::get(i32Type),
          rewriter.getStringAttr("&"), arr0S);
      operands.push_back(shiftPtr.getResult());

      if (globalAddrMap.count(rescaleSymbol)) {
        operands.push_back(globalAddrMap[rescaleSymbol]);
      } else {
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            r.getLoc(), lvalueType, FlatSymbolRefAttr::get(ctx, rescaleSymbol));
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            r.getLoc(), ptrType, rewriter.getStringAttr("&"),
            getGlobal.getResult());
        globalAddrMap[rescaleSymbol] = addrOf.getResult();
        operands.push_back(addrOf.getResult());
      }
      return success();
    };

    if (failed(addMultShiftOperands(rescale_lhs)))
      return failure();
    if (failed(addMultShiftOperands(rescale_temp)))
      return failure();
    if (failed(addMultShiftOperands(rescale_rhs)))
      return failure();
    if (failed(addMultShiftOperands(rescaleOp4)))
      return failure();

    auto itAdd = globalMap.find(result.getAsOpaquePointer());
    if (itAdd == globalMap.end())
      return addOp.emitError("Missing global mapping for add result");

    std::string addSymbol = itAdd->second.getSymName().str();
    if (globalAddrMap.count(addSymbol)) {
      operands.push_back(globalAddrMap[addSymbol]);
    } else {
      auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
          addOp.getLoc(), lvalueType, FlatSymbolRefAttr::get(ctx, addSymbol));
      auto addrOf = rewriter.create<emitc::ApplyOp>(addOp.getLoc(), ptrType,
                                                    rewriter.getStringAttr("&"),
                                                    getGlobal.getResult());
      globalAddrMap[addSymbol] = addrOf.getResult();
      operands.push_back(addrOf.getResult());
    }
    convParamMap[callee.str()] = {};
    rewriter.create<emitc::CallOpaqueOp>(addOp.getLoc(), TypeRange{}, callee,
                                         args, ArrayAttr(), operands);

    rewriter.eraseOp(rescale_lhs);
    rewriter.eraseOp(rescale_temp);
    rewriter.eraseOp(rescale_rhs);
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(rescaleOp4);
    return success();
  } else if (rescaleOp2 && rescaleOp3 && maxOp) {
    Value result = maxOp.getResult();
    Type resultType = result.getType();
    StringAttr callee = rewriter.getStringAttr("maximum");

    auto opaqueTensorType =
        emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
    auto lvalueType = emitc::LValueType::get(opaqueTensorType);
    auto ptrType = emitc::PointerType::get(opaqueTensorType);

    auto lhsInZp = rescaleOp1.getInputZpAttr();
    auto rhsInZp = rescaleOp2.getInputZpAttr();
    auto outOutZp = rescaleOp3.getOutputZpAttr();

    SmallVector<Attribute> argAttrs = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(2), rewriter.getIndexAttr(3),
        rewriter.getIndexAttr(4), rewriter.getIndexAttr(5),
        rewriter.getIndexAttr(6), rewriter.getIndexAttr(7),
        rewriter.getIndexAttr(8), rewriter.getIndexAttr(9),
        rewriter.getIndexAttr(10)};
    argAttrs.push_back(lhsInZp);
    argAttrs.push_back(rhsInZp);
    argAttrs.push_back(outOutZp);
    ArrayAttr args = rewriter.getArrayAttr(argAttrs);

    auto ctx = rewriter.getContext();
    auto i32Type = rewriter.getIntegerType(32);

    if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
      zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                     rewriter.getIndexType(),
                                                     rewriter.getIndexAttr(0));
    }

    SmallVector<Value> operands;
    {
      Value rescaleResult = rescaleOp1.getOperand();
      auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
      if (itRescale == globalMap.end())
        return rescaleOp1.emitError(
            "Missing global mapping for rescale lhs result");

      std::string rescaleSymbol = itRescale->second.getSymName().str();

      if (globalAddrMap.count(rescaleSymbol)) {
        operands.push_back(globalAddrMap[rescaleSymbol]);
      } else {
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp1.getLoc(), lvalueType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol));
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            rescaleOp1.getLoc(), ptrType, rewriter.getStringAttr("&"),
            getGlobal.getResult());
        globalAddrMap[rescaleSymbol] = addrOf.getResult();
        operands.push_back(addrOf.getResult());
      }
    }
    auto addMultShiftOperands = [&](tosa::RescaleOp r) -> LogicalResult {
      Value rescaleResult = r.getResult();
      auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
      if (itRescale == globalMap.end())
        return r.emitError("Missing global mapping for rescale result");

      std::string rescaleSymbol = itRescale->second.getSymName().str();

      // multiplier
      int64_t multiplierSize = r.getMultiplierAttr().size();
      auto multiplierArrayType =
          emitc::ArrayType::get({multiplierSize}, i32Type);
      auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
          r.getLoc(), multiplierArrayType,
          FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier"));
      auto arr0M = rewriter.create<emitc::SubscriptOp>(
          r.getLoc(), emitc::LValueType::get(i32Type), multiplierGlobal,
          SmallVector<Value, 1>{zeroIndex});
      auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
          r.getLoc(), emitc::PointerType::get(i32Type),
          rewriter.getStringAttr("&"), arr0M);
      operands.push_back(multiplierPtr.getResult());

      // shift
      int64_t shiftSize = r.getShiftAttr().size();
      auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
      auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
          r.getLoc(), shiftArrayType,
          FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift"));
      auto arr0S = rewriter.create<emitc::SubscriptOp>(
          r.getLoc(), emitc::LValueType::get(i32Type), shiftGlobal,
          SmallVector<Value, 1>{zeroIndex});
      auto shiftPtr = rewriter.create<emitc::ApplyOp>(
          r.getLoc(), emitc::PointerType::get(i32Type),
          rewriter.getStringAttr("&"), arr0S);
      operands.push_back(shiftPtr.getResult());

      if (globalAddrMap.count(rescaleSymbol)) {
        operands.push_back(globalAddrMap[rescaleSymbol]);
      } else {
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            r.getLoc(), lvalueType, FlatSymbolRefAttr::get(ctx, rescaleSymbol));
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            r.getLoc(), ptrType, rewriter.getStringAttr("&"),
            getGlobal.getResult());
        globalAddrMap[rescaleSymbol] = addrOf.getResult();
        operands.push_back(addrOf.getResult());
      }
      return success();
    };

    if (failed(addMultShiftOperands(rescaleOp1)))
      return failure();
    if (failed(addMultShiftOperands(rescaleOp2)))
      return failure();
    if (failed(addMultShiftOperands(rescaleOp3)))
      return failure();

    auto itMax = globalMap.find(result.getAsOpaquePointer());
    if (itMax == globalMap.end())
      return maxOp.emitError("Missing global mapping for maximum result");

    std::string maxSymbol = itMax->second.getSymName().str();
    if (globalAddrMap.count(maxSymbol)) {
      operands.push_back(globalAddrMap[maxSymbol]);
    } else {
      auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
          maxOp.getLoc(), lvalueType, FlatSymbolRefAttr::get(ctx, maxSymbol));
      auto addrOf = rewriter.create<emitc::ApplyOp>(maxOp.getLoc(), ptrType,
                                                    rewriter.getStringAttr("&"),
                                                    getGlobal.getResult());
      globalAddrMap[maxSymbol] = addrOf.getResult();
      operands.push_back(addrOf.getResult());
    }
    convParamMap[callee.str()] = {};
    rewriter.create<emitc::CallOpaqueOp>(maxOp.getLoc(), TypeRange{}, callee,
                                         args, ArrayAttr(), operands);

    rewriter.eraseOp(rescaleOp1);
    rewriter.eraseOp(rescaleOp2);
    rewriter.eraseOp(maxOp);
    rewriter.eraseOp(rescaleOp3);
  }
  return success();
}

// ReduceMax
LogicalResult
TosaToEmitCLowering::convertReduceMaxToEmitC(tosa::ReduceMaxOp reduceOp,
                                             PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("reduce_max");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  int64_t axisVal = reduceOp.getAxis();
  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getI64IntegerAttr(axisVal)});

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(reduceOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(reduceOp.getOperation(), rewriter);

  Value result = reduceOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return reduceOp.emitError("Missing global mapping for reduce_max result");
  }
  std::string symbolName = it->second.getSymName().str();

  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
      reduceOp.getLoc(), lvalueType,
      FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
  auto addrOf = rewriter.create<emitc::ApplyOp>(reduceOp.getLoc(), ptrType,
                                                rewriter.getStringAttr("&"),
                                                getGlobal.getResult());
  operands.push_back(addrOf.getResult());

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      reduceOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(reduceOp);
  llvm::errs() << "[DEBUG] reduce_max \n";
  return success();
}

// reduce_sum
LogicalResult
TosaToEmitCLowering::convertReduceSumToEmitC(tosa::ReduceSumOp reduceOp,
                                             PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("reduce_sum");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  int64_t axisVal = reduceOp.getAxis();
  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getI64IntegerAttr(axisVal)});

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(reduceOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(reduceOp.getOperation(), rewriter);

  Value result = reduceOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return reduceOp.emitError("Missing global mapping for reduce_sum result");
  }
  std::string symbolName = it->second.getSymName().str();

  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
      reduceOp.getLoc(), lvalueType,
      FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
  auto addrOf = rewriter.create<emitc::ApplyOp>(reduceOp.getLoc(), ptrType,
                                                rewriter.getStringAttr("&"),
                                                getGlobal.getResult());
  operands.push_back(addrOf.getResult());

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      reduceOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);

  rewriter.eraseOp(reduceOp);
  return success();
}

// sub
LogicalResult
TosaToEmitCLowering::convertSubToEmitC(tosa::SubOp subOp,
                                       PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("sub_float");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(2)});
  auto operands = operandsToGlobalPtrs(subOp.getOperation(), rewriter);

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(subOp.getResult().getType())});

  Value result = subOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return subOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        subOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(subOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      subOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(subOp);
  return success();
}

// mul
LogicalResult
TosaToEmitCLowering::convertMulToEmitC(tosa::MulOp mulOp,
                                       PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("mul_float");

  auto shiftAttr = mulOp->getAttrOfType<mlir::IntegerAttr>("shift");
  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args =
      rewriter.getArrayAttr({rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(2), shiftAttr});
  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(mulOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(mulOp.getOperation(), rewriter);

  if (Value shiftVal = mulOp.getShift()) {
    operands.push_back(shiftVal);
  }

  Value result = mulOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return mulOp.emitError("Missing global mapping for conv2d result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        mulOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(mulOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());

    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  auto newCall = rewriter.create<emitc::CallOpaqueOp>(
      mulOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
  rewriter.eraseOp(mulOp);
  return success();
}

// exp
LogicalResult
TosaToEmitCLowering::convertExpToEmitC(tosa::ExpOp expOp,
                                       PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("exp");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args = rewriter.getArrayAttr(
      {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)});

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(expOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(expOp.getOperation(), rewriter);

  Value result = expOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return expOp.emitError("Missing global mapping for exp result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        expOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));

    auto addrOf = rewriter.create<emitc::ApplyOp>(expOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  rewriter.create<emitc::CallOpaqueOp>(expOp.getLoc(), TypeRange{}, callee,
                                       args, ArrayAttr(), operands);

  rewriter.eraseOp(expOp);
  return success();
}

// reciprocal
LogicalResult
TosaToEmitCLowering::convertReciprocalToEmitC(tosa::ReciprocalOp recipOp,
                                              PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("reciprocal");

  auto opaqueTensorType =
      emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  ArrayAttr args = rewriter.getArrayAttr(
      {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)});

  ArrayAttr templateArgs =
      rewriter.getArrayAttr({TypeAttr::get(recipOp.getResult().getType())});

  auto operands = operandsToGlobalPtrs(recipOp.getOperation(), rewriter);

  Value result = recipOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end()) {
    return recipOp.emitError("Missing global mapping for reciprocal result");
  }
  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        recipOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(rewriter.getContext(), symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(recipOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  convParamMap[callee.str()] = {};

  rewriter.create<emitc::CallOpaqueOp>(recipOp.getLoc(), TypeRange{}, callee,
                                       args, ArrayAttr(), operands);

  rewriter.eraseOp(recipOp);
  return success();
}

// reshape
LogicalResult
TosaToEmitCLowering::convertReshapeToEmitC(tosa::ReshapeOp reshapeOp,
                                           PatternRewriter &rewriter) {
  StringAttr callee = rewriter.getStringAttr("reshape_s8");

  auto ctx = rewriter.getContext();
  auto opaqueTensorType = emitc::OpaqueType::get(ctx, "Tensor");
  auto lvalueType = emitc::LValueType::get(opaqueTensorType);
  auto ptrType = emitc::PointerType::get(opaqueTensorType);

  auto i32Type = rewriter.getI32Type();

  if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
    zeroIndex = rewriter.create<emitc::ConstantOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
  }
  ArrayAttr args = rewriter.getArrayAttr(
      {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)});
  auto operands = operandsToGlobalPtrs(reshapeOp.getOperation(), rewriter);

  Value result = reshapeOp.getResult();
  auto it = globalMap.find(result.getAsOpaquePointer());
  if (it == globalMap.end())
    return reshapeOp.emitError("Missing global mapping for reshape result");

  std::string symbolName = it->second.getSymName().str();

  if (globalAddrMap.count(symbolName)) {
    operands.push_back(globalAddrMap[symbolName]);
  } else {
    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        reshapeOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(ctx, symbolName));
    auto addrOf = rewriter.create<emitc::ApplyOp>(reshapeOp.getLoc(), ptrType,
                                                  rewriter.getStringAttr("&"),
                                                  getGlobal.getResult());
    globalAddrMap[symbolName] = addrOf.getResult();
    operands.push_back(addrOf.getResult());
  }

  std::string shapeSymbol = symbolName + "_new_shape";
  auto shapeGlobal = rewriter.create<emitc::GetGlobalOp>(
      reshapeOp.getLoc(),
      emitc::ArrayType::get({(int64_t)reshapeOp.getNewShape().size()}, i32Type),
      FlatSymbolRefAttr::get(ctx, shapeSymbol));

  auto arr0 = rewriter.create<emitc::SubscriptOp>(
      reshapeOp.getLoc(), emitc::LValueType::get(i32Type), shapeGlobal,
      SmallVector<Value, 1>{zeroIndex});
  auto shapePtr = rewriter.create<emitc::ApplyOp>(
      reshapeOp.getLoc(), emitc::PointerType::get(i32Type),
      rewriter.getStringAttr("&"), arr0);

  operands.push_back(shapePtr.getResult());

  convParamMap[callee.str()] = {};
  rewriter.create<emitc::CallOpaqueOp>(reshapeOp.getLoc(), TypeRange{}, callee,
                                       ArrayAttr(), ArrayAttr(), operands);

  rewriter.eraseOp(reshapeOp);
  return success();
}
