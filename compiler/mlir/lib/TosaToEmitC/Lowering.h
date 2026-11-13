#ifndef TINYGEN_TOSA_TO_EMITC_LOWERING_H
#define TINYGEN_TOSA_TO_EMITC_LOWERING_H

#include "MemoryPlanner/TensorGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tinygen {
using namespace mlir;
// One instance per module. No MLIR handles survive a compilation.
struct TosaToEmitCLowering {
  std::map<void *, mlir::emitc::GlobalOp> globalMap;
  std::map<std::string, mlir::Value> globalAddrMap;
  mlir::Value zeroIndex;
  std::map<std::string, std::vector<int64_t>> convParamMap;

  mlir::LogicalResult run(mlir::ModuleOp module, const std::string &opsFile);
  std::unique_ptr<TensorGraph> buildTensorGraph(mlir::func::FuncOp main);
  SmallVector<Value, 4> operandsToGlobalPtrs(Operation *op,
                                             PatternRewriter &rewriter);
  LogicalResult convertConstToEmitC(tosa::ConstOp constOp,
                                    PatternRewriter &rewriter);
  LogicalResult convertConv2DToEmitC(tosa::Conv2DOp convOp,
                                     PatternRewriter &rewriter);
  LogicalResult convertDepthwiseConv2DToEmitC(tosa::DepthwiseConv2DOp convOp,
                                              PatternRewriter &rewriter);
  LogicalResult convertMax_pool2DToEmitC(tosa::MaxPool2dOp max_poolOp,
                                         PatternRewriter &rewriter);
  LogicalResult convertAvg_pool2DToEmitC(tosa::AvgPool2dOp avg_poolOp,
                                         PatternRewriter &rewriter);
  LogicalResult convertPadToEmitC(tosa::PadOp padOp, PatternRewriter &rewriter);
  LogicalResult convertConstShapeToEmitC(tosa::ConstShapeOp constShapeOp,
                                         PatternRewriter &rewriter);
  LogicalResult convertConcatToEmitC(tosa::ConcatOp concatOp,
                                     PatternRewriter &rewriter);
  LogicalResult convertCastToEmitC(tosa::CastOp castOp,
                                   PatternRewriter &rewriter);
  LogicalResult
  convertTransposeConv2DToEmitC(tosa::TransposeConv2DOp transposeConvOp,
                                PatternRewriter &rewriter);
  LogicalResult convertTransposeToEmitC(tosa::TransposeOp transposeOp,
                                        PatternRewriter &rewriter);
  LogicalResult convertAddToEmitC(tosa::AddOp addOp, PatternRewriter &rewriter);
  LogicalResult convertRescaleToEmitC(tosa::RescaleOp rescaleOp1,
                                      PatternRewriter &rewriter);
  LogicalResult convertReduceMaxToEmitC(tosa::ReduceMaxOp reduceOp,
                                        PatternRewriter &rewriter);
  LogicalResult convertReduceSumToEmitC(tosa::ReduceSumOp reduceOp,
                                        PatternRewriter &rewriter);
  LogicalResult convertSubToEmitC(tosa::SubOp subOp, PatternRewriter &rewriter);
  LogicalResult convertMulToEmitC(tosa::MulOp mulOp, PatternRewriter &rewriter);
  LogicalResult convertExpToEmitC(tosa::ExpOp expOp, PatternRewriter &rewriter);
  LogicalResult convertReciprocalToEmitC(tosa::ReciprocalOp recipOp,
                                         PatternRewriter &rewriter);
  LogicalResult convertReshapeToEmitC(tosa::ReshapeOp reshapeOp,
                                      PatternRewriter &rewriter);
};
} // namespace tinygen

#endif
