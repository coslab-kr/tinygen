#include "Passes.h"
#include <optional>
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <fstream>
#include "mlir/IR/BuiltinTypes.h"

#include "MemoryPlanner/TensorGraph.cpp"
#include "MemoryPlanner/LivenessAnalysis.cpp"
#include "MemoryPlanner/MemoryPlanner.cpp"
#include "MemoryPlanner/TensorNode.cpp"

using namespace mlir;


namespace {
    static int id = 0;
    static std::map<void* , emitc::GlobalOp> globalMap;
    static std::map<std::string, mlir::Value> globalAddrMap;
    static mlir::Value zeroIndex = nullptr;
    std::map<std::string, std::vector<int64_t>> convParamMap;

    SmallVector<Value, 4> operandsToGlobalPtrs(Operation *op, PatternRewriter &rewriter) {
        
        MLIRContext *ctx = rewriter.getContext();
        Location loc = op->getLoc();
        SmallVector<Value,4 > operands;

        for(Value operand : op->getOperands()){
            auto it = globalMap.find(operand.getAsOpaquePointer());
            if(it == globalMap.end()){
                op->emitError()<<"Operand has no matching global mapping.\n";
                continue;
            }

            std::string symbolName = it->second.getSymName().str();
            
            if (globalAddrMap.count(symbolName)){
                operands.push_back(globalAddrMap[symbolName]);
                continue;
            }
            auto tensorTy = emitc::OpaqueType::get(ctx, "Tensor");
            auto ptrTy = emitc::PointerType::get(tensorTy);
            auto lvalueTy = emitc::LValueType::get(tensorTy);

            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                loc,
                lvalueTy,
                FlatSymbolRefAttr::get(ctx, symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                loc,
                ptrTy,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        return operands;
    }

    //const
    LogicalResult convertConstToEmitC(tosa::ConstOp constOp, PatternRewriter &rewriter) {
        auto *context = constOp.getContext();
        auto result = constOp.getResult();
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(constOp.getResult().getType());
        if (!resultType) { return failure(); }
        
        auto it = globalMap.find(result.getAsOpaquePointer());
        if(it == globalMap.end()){
            constOp.emitError("Global tensor not found in map for this SSA value");
            return failure();
        }
        
        std::string varName = it->second.getSymName().str();
        rewriter.eraseOp(constOp);
        return success();
    }

    //conv2d
    LogicalResult convertConv2DToEmitC(tosa::Conv2DOp convOp, PatternRewriter &rewriter) {
        
        if (!convOp->hasOneUse())
          return failure();
        auto rescaleOp = dyn_cast<tosa::RescaleOp>(*convOp->getResult(0).user_begin());
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

        if(filter_size == filter_size2){
            if(filter_size == 1){
                if((stride[0] == 1) && (stride[1] == 1)){
                    callee = rewriter.getStringAttr("conv2d_f1x1_s1x1_s8");
                }else{
                    callee = rewriter.getStringAttr("conv2d_f1x1_s8");
                }
            }
            else if(filter_size == 3){
                callee = rewriter.getStringAttr("conv2d_f3x3_s8");
            }else{
                callee = rewriter.getStringAttr("conv2d_fnxn_s8");
            }
        }
        else{
            callee = rewriter.getStringAttr("conv2d_fnxm_s8");
        }
          
        std::string varName = "t" + std::to_string(id++);
        
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        llvm::SmallVector<Attribute, 8> argAttrs  = {
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
        }
        else{
            int32_t zeroPoint = -1;
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
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
            zeroIndex = rewriter.create<emitc::ConstantOp>(
                rewriter.getUnknownLoc(),
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0)
            );
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
            rescaleOp.getLoc(),
            multiplierArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier")
        );
        auto arr0M = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            multiplierGlobal,
            SmallVector<Value,1>{zeroIndex}
        );
        auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0M)
        ;
        operands.push_back(multiplierPtr.getResult());

        // shift
        int64_t shiftSize = rescaleOp.getShiftAttr().size();
        auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
        auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp.getLoc(),
            shiftArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift")
        );
        auto arr0S = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            shiftGlobal,
            SmallVector<Value,1>{zeroIndex}
        );
        auto shiftPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0S
        );
        operands.push_back(shiftPtr.getResult());
        
        if (globalAddrMap.count(rescaleSymbol)) {
            operands.push_back(globalAddrMap[rescaleSymbol]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                rescaleOp.getLoc(), lvalueType,
                FlatSymbolRefAttr::get(ctx, rescaleSymbol)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                rescaleOp.getLoc(), ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
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
                convOp.getLoc(), 
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(),symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                convOp.getLoc(), 
                ptrType, 
                rewriter.getStringAttr("&"), 
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        auto padAttr = convOp.getPad();        
        auto strideAttr = convOp.getStride();  
        auto dilationAttr = convOp.getDilation(); 
        
        std::vector<int64_t> params = {
            padAttr[0],
            padAttr[2],
            strideAttr[0],
            strideAttr[1],
            dilationAttr[0],
            dilationAttr[1]
        };
        if(callee.str() == "conv2d_f1x1_s1x1_s8"){
            params.clear();
        }

        auto key = callee.str();
        
        if(callee.str() == "conv2d_fnxm_s8" || callee.str() == "conv2d_fnxn_s8"){
            key = "conv2d_f3x3_s8";
            convParamMap[callee.str()] = {};
        }
        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            convOp.getLoc(),  
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );
        
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

        if (clampOp) rewriter.eraseOp(clampOp);
        rewriter.eraseOp(rescaleOp);
        rewriter.eraseOp(convOp);
        return success();
    }

    //depthwise_conv2d
    LogicalResult convertDepthwiseConv2DToEmitC(tosa::DepthwiseConv2DOp convOp,
                                                PatternRewriter &rewriter) {

        if (!convOp->hasOneUse())
            return failure();
        auto rescaleOp = dyn_cast<tosa::RescaleOp>(*convOp->getResult(0).user_begin());
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

        std::string varName = "t" + std::to_string(id++);
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
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
            int32_t zeroPoint = -1;
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
        }
        argAttrs.push_back(rescaleOp.getOutputZpAttr());

        ArrayAttr args = rewriter.getArrayAttr(argAttrs);
        ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(lvalueType)});
        auto operands = operandsToGlobalPtrs(convOp.getOperation(), rewriter);

        auto ctx = rewriter.getContext();
        auto i32Type = rewriter.getI32Type();

        if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
            zeroIndex = rewriter.create<emitc::ConstantOp>(
                rewriter.getUnknownLoc(),
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0)
            );
        }

        Value rescaleResult = rescaleOp.getResult();
        auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
        if (itRescale == globalMap.end())
            return convOp.emitError("Missing global mapping for rescale result");
        std::string rescaleSymbol = itRescale->second.getSymName().str();

        Value convResult = convOp.getResult();
        auto itConv = globalMap.find(convResult.getAsOpaquePointer());
        if (itConv == globalMap.end())
            return convOp.emitError("Missing global mapping for depthwise conv2d result");
        std::string convSymbol = itConv->second.getSymName().str();

        // multiplier
        int64_t multiplierSize = rescaleOp.getMultiplierAttr().size();
        auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
        auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp.getLoc(),
            multiplierArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier")
        );
        auto arr0M = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            multiplierGlobal,
            SmallVector<Value, 1>{zeroIndex}
        );
        auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0M
        );
        operands.push_back(multiplierPtr.getResult());

        // shift
        int64_t shiftSize = rescaleOp.getShiftAttr().size();
        auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
        auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp.getLoc(),
            shiftArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift")
        );
        auto arr0S = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            shiftGlobal,
            SmallVector<Value, 1>{zeroIndex}
        );
        auto shiftPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0S
        );
        operands.push_back(shiftPtr.getResult());

        if (globalAddrMap.count(rescaleSymbol)) {
            operands.push_back(globalAddrMap[rescaleSymbol]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                rescaleOp.getLoc(), 
                lvalueType,
                FlatSymbolRefAttr::get(ctx, rescaleSymbol)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                rescaleOp.getLoc(), 
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[rescaleSymbol] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        Value finalResult = rescaleOp.getResult();
        auto it = globalMap.find(finalResult.getAsOpaquePointer());
        if (it == globalMap.end())
            return convOp.emitError("Missing global mapping for fused depthwise conv result");
        std::string symbolName = it->second.getSymName().str();

        if (globalAddrMap.count(symbolName)) {
            operands.push_back(globalAddrMap[symbolName]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                convOp.getLoc(), lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
                );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                convOp.getLoc(), ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
                );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        auto padAttr = convOp.getPad();
        auto strideAttr = convOp.getStride();
        auto dilationAttr = convOp.getDilation();

        std::vector<int64_t> params = {
            padAttr[0],
            padAttr[2],
            strideAttr[0],
            strideAttr[1],
            dilationAttr[0],
            dilationAttr[1]
        };
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
    LogicalResult convertMax_pool2DToEmitC(tosa::MaxPool2dOp max_poolOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("max_pool_2d");
        
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getI64TensorAttr(max_poolOp.getPad()),
            rewriter.getI64TensorAttr(max_poolOp.getStride()),
            rewriter.getI64TensorAttr(max_poolOp.getKernel()),
        });
        
        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(max_poolOp.getResult().getType())
        });
      
        auto operands = operandsToGlobalPtrs(max_poolOp.getOperation(), rewriter);
        Value result = max_poolOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return max_poolOp.emitError("Missing global mapping for conv2d result");
        }
        std::string symbolName = it->second.getSymName().str();

        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            max_poolOp.getLoc(),
            lvalueType,
            FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
        );
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            max_poolOp.getLoc(),
            ptrType,
            rewriter.getStringAttr("&"),
            getGlobal.getResult()
        );
        operands.push_back(addrOf.getResult());
        
        auto padAttr = max_poolOp.getPad();   
        auto strideAttr = max_poolOp.getStride(); 

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            max_poolOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );
        
        std::vector<int64_t> params = {
            padAttr[0],  
            padAttr[2],  
            strideAttr[0],
            strideAttr[1],
            rewriter.getI64IntegerAttr(1).getInt(), 
            rewriter.getI64IntegerAttr(1).getInt() 
        };

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

    //avg_pool2d
    LogicalResult convertAvg_pool2DToEmitC(tosa::AvgPool2dOp avg_poolOp,
                                          PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("avg_pool_2d");

        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getI64TensorAttr(avg_poolOp.getPad()),
            rewriter.getI64TensorAttr(avg_poolOp.getStride()),
            rewriter.getI64TensorAttr(avg_poolOp.getKernel()),
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(avg_poolOp.getResult().getType())
        });

        auto operands = operandsToGlobalPtrs(avg_poolOp.getOperation(), rewriter);

        Value result = avg_poolOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return avg_poolOp.emitError("Missing global mapping for avg_pool2d result");
        }
        std::string symbolName = it->second.getSymName().str();

        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            avg_poolOp.getLoc(),
            lvalueType,
            FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
        );

        auto addrOf = rewriter.create<emitc::ApplyOp>(
            avg_poolOp.getLoc(),
            ptrType,
            rewriter.getStringAttr("&"),
            getGlobal.getResult()
        );
        operands.push_back(addrOf.getResult());

        auto padAttr = avg_poolOp.getPad();      
        auto strideAttr = avg_poolOp.getStride();

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            avg_poolOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );

        std::vector<int64_t> params = {
            padAttr[0],      
            padAttr[2],     
            strideAttr[0], 
            strideAttr[1],
            rewriter.getI64IntegerAttr(1).getInt(), 
            rewriter.getI64IntegerAttr(1).getInt() 
        };

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
    LogicalResult convertPadToEmitC(tosa::PadOp padOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("pad_s8");
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2),
            rewriter.getIndexAttr(3)
        });
        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(padOp.getResult().getType())
        });

        auto operands = operandsToGlobalPtrs(padOp.getOperation(), rewriter);

        Value result = padOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return padOp.emitError("Missing global mapping for conv2d result");
        }
        std::string symbolName = it->second.getSymName().str();
        
        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            padOp.getLoc(),
            lvalueType,
            FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
        );
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            padOp.getLoc(),
            ptrType,
            rewriter.getStringAttr("&"),
            getGlobal.getResult()
        );
        operands.push_back(addrOf.getResult());

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            padOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args,  
            ArrayAttr(),
            operands
        ); 
        convParamMap[callee.str()] = {};
        rewriter.eraseOp(padOp);
        return success();
    }

    //const_shape
    LogicalResult convertConstShapeToEmitC(tosa::ConstShapeOp constShapeOp, PatternRewriter &rewriter) {
        auto *context = constShapeOp.getContext();
        auto result = constShapeOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if(it == globalMap.end()){
            constShapeOp.emitError("Global tensor not found in map for this SSA value");
            return failure();
        }
        rewriter.eraseOp(constShapeOp);
        return success();
    }

    //concat
    LogicalResult convertConcatToEmitC(tosa::ConcatOp concatOp, PatternRewriter &rewriter) {
          
        Location loc = concatOp.getLoc();
        IntegerAttr axisAttr = concatOp.getAxisAttr();
        int64_t axis = axisAttr.getInt();
        
        StringAttr callee;
        if(axis == 0){
            callee =  rewriter.getStringAttr("concat_axis0");
        }
        else if(axis == 1){
            callee =  rewriter.getStringAttr("concat_axis1");
        }
        else if(axis == 2){
            callee =  rewriter.getStringAttr("concat_axis2");
        }
        else{
            callee =  rewriter.getStringAttr("concat_axis3");
        }
          
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2),
            concatOp.getAxisAttr()
        });
        
        ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(lvalueType)});
        
        auto now_concatOp = concatOp;

        SmallVector<Value, 4> operands_concat;
        
        for(Value operand : concatOp.getOperands()){
            auto it = globalMap.find(operand.getAsOpaquePointer());
            if (it == globalMap.end()) {
                concatOp.emitError() << "Missing global tensor for operand";
                return failure();
            }
            std::string symbolName = it->second.getSymName().str();
                
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                loc, 
                lvalueType, 
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                loc, 
                ptrType, 
                rewriter.getStringAttr("&"), 
                getGlobal.getResult()
            );
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
                concatOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                concatOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );

            globalAddrMap[symbolName] = addrOf.getResult();
            operands_concat.push_back(addrOf.getResult());
        }

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            loc, 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands_concat
        );
        convParamMap[callee.str()] = {};
        rewriter.eraseOp(concatOp); 
        return success();
    }

    //cast
    LogicalResult convertCastToEmitC(tosa::CastOp castOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("cast");
      
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
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
                llvm::errs() << "[DEBUG] Cast direction: int8 → float32\n";
                quant_check = 0;
            }
            else if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inElemType)) {
                if (quantType.getStorageTypeIntegralWidth() == 8 && outElemType.isF32()) {
                    llvm::errs() << "[DEBUG] Cast direction: quantized int8 → float32 (dequant)\n";
                    quant_check = 0;
                }
            }
            else if (inElemType.isF32() && outElemType.isInteger(8)) {
                llvm::errs() << "[DEBUG] Cast direction: float32 → int8\n";
                quant_check = 1;
            }
            else if (inElemType.isF32()) {
                if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType)) {
                    if (quantType.getStorageTypeIntegralWidth() == 8) {
                        llvm::errs() << "[DEBUG] Cast direction: float32 → quantized int8 (quant)\n";
                        quant_check = 1;
                    }
                }
            }
            else {
                llvm::errs() << "[DEBUG] Cast direction: other type conversion\n";
                quant_check = -1;
            }

        } else {
            llvm::errs() << "[DEBUG] Non-tensor cast type, skipping\n";
            quant_check = -1;
        }

        SmallVector<mlir::Attribute, 4> argsVec = {
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getI32IntegerAttr(quant_check) 
        };
        ArrayAttr args = rewriter.getArrayAttr(argsVec);

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(castOp.getResult().getType())
        });
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
                castOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                castOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );

            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }
        
        convParamMap[callee.str()] = {};
        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            castOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(),
            operands
        );
        rewriter.eraseOp(castOp);
        return success();
    }

    //transpose_conv2d
    LogicalResult convertTransposeConv2DToEmitC(tosa::TransposeConv2DOp transposeConvOp, PatternRewriter &rewriter) {
        
        if (!transposeConvOp->hasOneUse())
          return failure();
        auto rescaleOp = dyn_cast<tosa::RescaleOp>(*transposeConvOp->getResult(0).user_begin());
        if (!rescaleOp)
          return failure();
        
        auto strideAttr = transposeConvOp.getStride();
        
        auto inputType = mlir::cast<mlir::ShapedType>(transposeConvOp.getInput().getType());
        auto inputShape = inputType.getShape();
        int64_t input_ch = inputShape[3];
        
        StringAttr callee;
            
        if(strideAttr[0] <= 2 && strideAttr[1] <=2 && input_ch >16){
            callee = rewriter.getStringAttr("transpose_conv2d_s1_s2_s8");
        }else{
            callee = rewriter.getStringAttr("transpose_conv2d_s8");
        }
      
        std::string varName = "t" + std::to_string(id++);
        
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
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
            rewriter.getI64TensorAttr(transposeConvOp.getOutShape())
        };
        auto quantInfo = transposeConvOp.getQuantizationInfo();

        if (quantInfo.has_value()) {
            argAttrs.push_back(rewriter.getI32IntegerAttr(quantInfo->getInputZp()));
        }
        else{
            int32_t zeroPoint = -1;
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
            argAttrs.push_back(rewriter.getI32IntegerAttr(zeroPoint));
        }

        argAttrs.push_back(rescaleOp.getOutputZpAttr());

        ArrayAttr args = rewriter.getArrayAttr(argAttrs);

        auto operands = operandsToGlobalPtrs(transposeConvOp.getOperation(), rewriter);
        
        auto ctx = rewriter.getContext();
        auto i32Type = rewriter.getI32Type();
        
        Value transResult = transposeConvOp.getResult();
        auto itTrans = globalMap.find(transResult.getAsOpaquePointer());
        if (itTrans == globalMap.end())
            return transposeConvOp.emitError("Missing global mapping for transpose result");
        std::string transSymbol = itTrans->second.getSymName().str();

        if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
            zeroIndex = rewriter.create<emitc::ConstantOp>(
                rewriter.getUnknownLoc(),
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0)
            );
        }
        Value rescaleResult = rescaleOp.getResult();
        auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
        if (itRescale == globalMap.end())
            return transposeConvOp.emitError("Missing global mapping for rescale result");
        std::string rescaleSymbol = itRescale->second.getSymName().str();

        //  multiplier 
        int64_t multiplierSize = rescaleOp.getMultiplierAttr().size();
        auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
        auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp.getLoc(),
            multiplierArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier")
        );
        auto arr0M = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            multiplierGlobal,
            SmallVector<Value,1>{zeroIndex}
        );
        auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0M
        );
        operands.push_back(multiplierPtr.getResult());

        //shift
        int64_t shiftSize = rescaleOp.getShiftAttr().size();
        auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
        auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
            rescaleOp.getLoc(),
            shiftArrayType,
            FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift")
        );
        auto arr0S = rewriter.create<emitc::SubscriptOp>(
            rescaleOp.getLoc(),
            emitc::LValueType::get(i32Type),
            shiftGlobal,
            SmallVector<Value,1>{zeroIndex}
        );
        auto shiftPtr = rewriter.create<emitc::ApplyOp>(
            rescaleOp.getLoc(),
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"),
            arr0S
        );
        operands.push_back(shiftPtr.getResult());

        // output tensor pointer
        Value finalResult = rescaleOp.getResult();
        auto it = globalMap.find(finalResult.getAsOpaquePointer());
        if (it == globalMap.end())
          return transposeConvOp.emitError("Missing global mapping for fused transpose result");
        std::string symbolName = it->second.getSymName().str();
        if (globalAddrMap.count(symbolName)) {
            operands.push_back(globalAddrMap[symbolName]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                transposeConvOp.getLoc(), 
                lvalueType,
                FlatSymbolRefAttr::get(ctx, symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                transposeConvOp.getLoc(), 
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }
        auto padAttr = transposeConvOp.getOutPad(); 
        auto filterType = mlir::cast<mlir::ShapedType>(transposeConvOp.getFilter().getType());
        auto filterShape = filterType.getShape(); 

        int64_t filter_h = filterShape[1];
        int64_t filter_w = filterShape[2];

        int64_t out_pad_h = padAttr[0];   
        int64_t out_pad_w = padAttr[2];  

        int64_t padding_h = filter_h - 1 - out_pad_h;
        int64_t padding_w = filter_w - 1 - out_pad_w;    

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            transposeConvOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(),
            operands
        );
        
        std::vector<int64_t> params = {
            padding_h,
            padding_w,
            rewriter.getI64IntegerAttr(1).getInt(),
            rewriter.getI64IntegerAttr(1).getInt(),
            rewriter.getI64IntegerAttr(1).getInt(),
            rewriter.getI64IntegerAttr(1).getInt()
        };
        auto key = callee.str();
        
        if( callee.str() == "transpose_conv2d_s1_s2_s8"){
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
    LogicalResult convertTransposeToEmitC(tosa::TransposeOp transposeOp,
                                          PatternRewriter &rewriter) {
        Value result = transposeOp.getResult();
        Type resultType = result.getType();
        StringAttr callee = rewriter.getStringAttr("transpose");

        llvm::errs() << "[DEBUG] Selected callee: " << callee.getValue() << "\n";

        std::string varName = "t" + std::to_string(id++);
        
        if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
            zeroIndex = rewriter.create<emitc::ConstantOp>(
                rewriter.getUnknownLoc(),
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0));
        }
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        auto operands = operandsToGlobalPtrs(transposeOp.getOperation(), rewriter);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2)
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(transposeOp.getResult().getType())
        });

        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return transposeOp.emitError("Missing global mapping for transpose result");
        }
        std::string symbolName = it->second.getSymName().str();

        if (globalAddrMap.count(symbolName)) {
            operands.push_back(globalAddrMap[symbolName]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                transposeOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                transposeOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );

            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        convParamMap[callee.str()] = {};

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            transposeOp.getLoc(),
            TypeRange{},
            callee,
            args,          
            ArrayAttr(),  
            operands
        );
        rewriter.eraseOp(transposeOp);
        return success();
    }


    //add
    LogicalResult convertAddToEmitC(tosa::AddOp addOp, PatternRewriter &rewriter){
        Value result = addOp.getResult();
        Type resultType = addOp.getResult().getType();
        StringAttr callee = rewriter.getStringAttr("add_float");
        
        llvm::errs() << "[DEBUG] Selected callee: " << callee.getValue() << "\n";

        std::string varName = "t" + std::to_string(id++);
        
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        auto operands = operandsToGlobalPtrs(addOp.getOperation(), rewriter);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2)
        });
        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(addOp.getResult().getType())
        });

        
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return addOp.emitError("Missing global mapping for conv2d result");
        }
        std::string symbolName = it->second.getSymName().str();
        
        if (globalAddrMap.count(symbolName)) {
            operands.push_back(globalAddrMap[symbolName]);
        } else {
            auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                addOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                addOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );

            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        convParamMap[callee.str()] = {};
        
        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            addOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );
        rewriter.eraseOp(addOp);
        return success();
    }

    //add_c & Maximum
    LogicalResult convertRescaleToEmitC(tosa::RescaleOp rescaleOp1, PatternRewriter &rewriter) {
        auto addUser = rescaleOp1->getResult(0).use_empty() ? nullptr : *rescaleOp1->getResult(0).user_begin();
        auto addOp   = dyn_cast_or_null<tosa::AddOp>(addUser);
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
            Operation *user1 = (*rescaleOp1->getResult(0).user_begin());
            rescaleOp2 = dyn_cast<tosa::RescaleOp>(user1);
            if (!rescaleOp2) {
                return failure();
            }

            // Check whether the user of rescale2 is add
            addOp = dyn_cast<tosa::AddOp>(*rescaleOp2->getResult(0).user_begin());
            if (!addOp) {
                return failure();
            }

            // The user of add is rescale4
            auto rescaleUser = addOp->getResult(0).use_empty() ? nullptr : *addOp->getResult(0).user_begin();
            rescaleOp4  = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
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
        }else if( addOp && !maxOp){
            check = 1;

            auto rescaleUser = addOp->getResult(0).use_empty() ? nullptr : *addOp->getResult(0).user_begin();
            rescaleOp4  = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
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
            rescale_lhs  = rescaleOp1;
            rescale_temp = rescaleOp3;
            rescale_rhs = rescaleOp2;

        }else if(maxOp){
            rescaleOp2 = maxOp.getOperand(0).getDefiningOp<tosa::RescaleOp>();
            auto rescaleUser = maxOp->getResult(0).use_empty() ? nullptr : *maxOp->getResult(0).user_begin();
            rescaleOp3  = dyn_cast_or_null<tosa::RescaleOp>(rescaleUser);
        }
        
        if (rescaleOp2 && rescaleOp3 && addOp && rescaleOp4 && !maxOp) {
            Value result = addOp.getResult();
            Type resultType = result.getType();
            StringAttr callee = rewriter.getStringAttr("add_s8");
            std::string varName = "t" + std::to_string(id++);

            auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
            auto lvalueType = emitc::LValueType::get(opaqueTensorType);
            auto ptrType = emitc::PointerType::get(opaqueTensorType);

            // --- ZP attributes (rescale1, rescale2, rescale3, rescale4) ---
            auto lhsInZp   = rescale_lhs.getInputZpAttr();
            auto tempInZp   = rescale_temp.getInputZpAttr();
            auto rhsInZp2  = rescale_rhs.getInputZpAttr();
            auto outOutZp  = rescaleOp4.getOutputZpAttr();
            
            SmallVector<Attribute> argAttrs = {
                rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(2),
                rewriter.getIndexAttr(3),
                rewriter.getIndexAttr(4),
                rewriter.getIndexAttr(5),
                rewriter.getIndexAttr(6),
                rewriter.getIndexAttr(7),
                rewriter.getIndexAttr(8),
                rewriter.getIndexAttr(9),
                rewriter.getIndexAttr(10),
                rewriter.getIndexAttr(11),
                rewriter.getIndexAttr(12),
                rewriter.getIndexAttr(13),
                rewriter.getIndexAttr(14)
            };
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
                  zeroIndex = rewriter.create<emitc::ConstantOp>(
                      rewriter.getUnknownLoc(), rewriter.getIndexType(),
                      rewriter.getIndexAttr(0));
            }     

            SmallVector<Value> operands;
          
            {
              Value rescaleResult = rescale_lhs.getOperand();
              auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
              if (itRescale == globalMap.end())
              return rescaleOp1.emitError("Missing global mapping for rescale lhs result");

                std::string rescaleSymbol = itRescale->second.getSymName().str();

              if (globalAddrMap.count(rescaleSymbol)) {
                  operands.push_back(globalAddrMap[rescaleSymbol]);
              } else {
                  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                      rescaleOp1.getLoc(), 
                      lvalueType,
                      FlatSymbolRefAttr::get(ctx, rescaleSymbol)
                  );
          
                  auto addrOf = rewriter.create<emitc::ApplyOp>(
                      rescaleOp1.getLoc(), 
                      ptrType,
                      rewriter.getStringAttr("&"), 
                      getGlobal.getResult()
                  );

                  globalAddrMap[rescaleSymbol] = addrOf.getResult();
                  operands.push_back(addrOf.getResult());
              } 
            }
            //input2: either rescaleOp3 or rescale_temp depending on 'check'
            {
              Value rescaleResult = rescale_rhs.getOperand();
              auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
              if (itRescale == globalMap.end())
                return rescaleOp1.emitError("Missing global mapping for rhs rescale result");

              std::string rescaleSymbol = itRescale->second.getSymName().str();
              if (globalAddrMap.count(rescaleSymbol)) {
                  operands.push_back(globalAddrMap[rescaleSymbol]);
              } else {
                auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                    rescaleOp1.getLoc(), 
                    lvalueType,
                    FlatSymbolRefAttr::get(ctx, rescaleSymbol)
                );

                auto addrOf = rewriter.create<emitc::ApplyOp>(
                    rescaleOp1.getLoc(), 
                    ptrType,
                    rewriter.getStringAttr("&"), 
                    getGlobal.getResult()
                );
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

            if (failed(addMultShiftOperands(rescale_lhs))) return failure();
            if (failed(addMultShiftOperands(rescale_temp))) return failure();
            if (failed(addMultShiftOperands(rescale_rhs))) return failure();
            if (failed(addMultShiftOperands(rescaleOp4))) return failure();

            auto itAdd = globalMap.find(result.getAsOpaquePointer());
            if (itAdd == globalMap.end())
              return addOp.emitError("Missing global mapping for add result");

            std::string addSymbol = itAdd->second.getSymName().str();
            if (globalAddrMap.count(addSymbol)) {
                operands.push_back(globalAddrMap[addSymbol]);
            } else {
                auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                    addOp.getLoc(), 
                    lvalueType, 
                    FlatSymbolRefAttr::get(ctx, addSymbol)
                );
                auto addrOf = rewriter.create<emitc::ApplyOp>(
                    addOp.getLoc(), 
                    ptrType, 
                    rewriter.getStringAttr("&"),
                    getGlobal.getResult()
                );
                globalAddrMap[addSymbol] = addrOf.getResult();
                operands.push_back(addrOf.getResult());
            }
            convParamMap[callee.str()] = {};
            rewriter.create<emitc::CallOpaqueOp>(
                addOp.getLoc(), 
                TypeRange{}, 
                callee, 
                args, 
                ArrayAttr(), 
                operands
            );

            rewriter.eraseOp(rescale_lhs);
            rewriter.eraseOp(rescale_temp);
            rewriter.eraseOp(rescale_rhs);
            rewriter.eraseOp(addOp);
            rewriter.eraseOp(rescaleOp4);
            return success();
        }
        else if(rescaleOp2 && rescaleOp3 && maxOp) {
            Value result = maxOp.getResult();
            Type resultType = result.getType();
            StringAttr callee = rewriter.getStringAttr("maximum");
            std::string varName = "t" + std::to_string(id++);
        
            auto opaqueTensorType =
                    emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
            auto lvalueType = emitc::LValueType::get(opaqueTensorType);
            auto ptrType = emitc::PointerType::get(opaqueTensorType);
        
            auto lhsInZp  = rescaleOp1.getInputZpAttr();
            auto rhsInZp  = rescaleOp2.getInputZpAttr();
            auto outOutZp = rescaleOp3.getOutputZpAttr();
            
            SmallVector<Attribute> argAttrs = {
                rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(2),
                rewriter.getIndexAttr(3),
                rewriter.getIndexAttr(4),
                rewriter.getIndexAttr(5),
                rewriter.getIndexAttr(6),
                rewriter.getIndexAttr(7),
                rewriter.getIndexAttr(8),
                rewriter.getIndexAttr(9),
                rewriter.getIndexAttr(10)
            };
            argAttrs.push_back(lhsInZp);
            argAttrs.push_back(rhsInZp);
            argAttrs.push_back(outOutZp);
            ArrayAttr args = rewriter.getArrayAttr(argAttrs);
        
            auto ctx = rewriter.getContext();
            auto i32Type = rewriter.getIntegerType(32);
        
            if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
                    zeroIndex = rewriter.create<emitc::ConstantOp>(
                        rewriter.getUnknownLoc(), 
                        rewriter.getIndexType(),
                        rewriter.getIndexAttr(0)
                    );
            }

            SmallVector<Value> operands;
            {
              Value rescaleResult = rescaleOp1.getOperand();
              auto itRescale = globalMap.find(rescaleResult.getAsOpaquePointer());
              if (itRescale == globalMap.end())
              return rescaleOp1.emitError("Missing global mapping for rescale lhs result");

              std::string rescaleSymbol = itRescale->second.getSymName().str();

              if (globalAddrMap.count(rescaleSymbol)) {
                  operands.push_back(globalAddrMap[rescaleSymbol]);
              } else {
                  auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                      rescaleOp1.getLoc(), 
                      lvalueType,
                      FlatSymbolRefAttr::get(ctx, rescaleSymbol)
                  );
                  auto addrOf = rewriter.create<emitc::ApplyOp>(
                      rescaleOp1.getLoc(), 
                      ptrType,
                      rewriter.getStringAttr("&"), 
                      getGlobal.getResult()
                  );
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
                auto multiplierArrayType = emitc::ArrayType::get({multiplierSize}, i32Type);
                auto multiplierGlobal = rewriter.create<emitc::GetGlobalOp>(
                    r.getLoc(), 
                    multiplierArrayType,
                    FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_multiplier")
                );
                auto arr0M = rewriter.create<emitc::SubscriptOp>(
                    r.getLoc(), 
                    emitc::LValueType::get(i32Type), 
                    multiplierGlobal,
                    SmallVector<Value, 1>{zeroIndex}
                );
                auto multiplierPtr = rewriter.create<emitc::ApplyOp>(
                    r.getLoc(), emitc::PointerType::get(i32Type),
                    rewriter.getStringAttr("&"), 
                    arr0M
                );
                operands.push_back(multiplierPtr.getResult());

                // shift
                int64_t shiftSize = r.getShiftAttr().size();
                auto shiftArrayType = emitc::ArrayType::get({shiftSize}, i32Type);
                auto shiftGlobal = rewriter.create<emitc::GetGlobalOp>(
                    r.getLoc(), 
                    shiftArrayType,
                    FlatSymbolRefAttr::get(ctx, rescaleSymbol + "_shift")
                );
                auto arr0S = rewriter.create<emitc::SubscriptOp>(
                    r.getLoc(), 
                    emitc::LValueType::get(i32Type),
                    shiftGlobal,
                    SmallVector<Value, 1>{zeroIndex}
                );
                auto shiftPtr = rewriter.create<emitc::ApplyOp>(
                    r.getLoc(), 
                    emitc::PointerType::get(i32Type),
                    rewriter.getStringAttr("&"), 
                    arr0S
                );
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

            if (failed(addMultShiftOperands(rescaleOp1))) return failure();
            if (failed(addMultShiftOperands(rescaleOp2))) return failure();
            if (failed(addMultShiftOperands(rescaleOp3))) return failure();
        
            auto itMax = globalMap.find(result.getAsOpaquePointer());
            if (itMax == globalMap.end())
              return maxOp.emitError("Missing global mapping for maximum result");
        
            std::string maxSymbol = itMax->second.getSymName().str();
            if (globalAddrMap.count(maxSymbol)) {
                operands.push_back(globalAddrMap[maxSymbol]);
            } else {
                auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
                    maxOp.getLoc(), lvalueType, FlatSymbolRefAttr::get(ctx, maxSymbol));
                auto addrOf = rewriter.create<emitc::ApplyOp>(
                    maxOp.getLoc(), ptrType, rewriter.getStringAttr("&"),
                        getGlobal.getResult());
                globalAddrMap[maxSymbol] = addrOf.getResult();
                operands.push_back(addrOf.getResult());
            }
            convParamMap[callee.str()] = {};
            rewriter.create<emitc::CallOpaqueOp>(
                maxOp.getLoc(), TypeRange{}, callee, args, ArrayAttr(), operands);
        
            rewriter.eraseOp(rescaleOp1);
            rewriter.eraseOp(rescaleOp2);
            rewriter.eraseOp(maxOp);
            rewriter.eraseOp(rescaleOp3);
        } 
        return success();
    }

    //ReduceMax
    LogicalResult convertReduceMaxToEmitC(tosa::ReduceMaxOp reduceOp,
                                          PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("reduce_max");
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        int64_t axisVal = reduceOp.getAxis();
        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getI64IntegerAttr(axisVal)
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(reduceOp.getResult().getType())
        });

        auto operands = operandsToGlobalPtrs(reduceOp.getOperation(), rewriter);
        
        Value result = reduceOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return reduceOp.emitError("Missing global mapping for reduce_max result");
        }
        std::string symbolName = it->second.getSymName().str();

        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            reduceOp.getLoc(),
            lvalueType,
            FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
        );
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            reduceOp.getLoc(),
            ptrType,
            rewriter.getStringAttr("&"),
            getGlobal.getResult()
        );
        operands.push_back(addrOf.getResult());

        convParamMap[callee.str()] = {};

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            reduceOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );
        rewriter.eraseOp(reduceOp);
        llvm::errs() <<"[DEBUG] reduce_max \n";
        return success();
    }

    //reduce_sum
    LogicalResult convertReduceSumToEmitC(tosa::ReduceSumOp reduceOp,
                                          PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("reduce_sum");
        std::string varName = "t" + std::to_string(id++);

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        int64_t axisVal = reduceOp.getAxis();
        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getI64IntegerAttr(axisVal) 
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(reduceOp.getResult().getType())
        });

        auto operands = operandsToGlobalPtrs(reduceOp.getOperation(), rewriter);

        Value result = reduceOp.getResult();
        auto it = globalMap.find(result.getAsOpaquePointer());
        if (it == globalMap.end()) {
            return reduceOp.emitError("Missing global mapping for reduce_sum result");
        }
        std::string symbolName = it->second.getSymName().str();

        auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
            reduceOp.getLoc(),
            lvalueType,
            FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
        );
        auto addrOf = rewriter.create<emitc::ApplyOp>(
            reduceOp.getLoc(),
            ptrType,
            rewriter.getStringAttr("&"),
            getGlobal.getResult()
        );
        operands.push_back(addrOf.getResult());
        
        convParamMap[callee.str()] = {};

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            reduceOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );

        rewriter.eraseOp(reduceOp);
        return success();
    }


    //sub
    LogicalResult convertSubToEmitC(tosa::SubOp subOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("subi_float");
        std::string varName = "t" + std::to_string(id++);
        
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2)
        });
        auto operands = operandsToGlobalPtrs(subOp.getOperation(), rewriter);

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(subOp.getResult().getType())
        });

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
                subOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                subOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }
        
        convParamMap[callee.str()] = {};

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            subOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );
        rewriter.eraseOp(subOp);
        return success();
    }

    //mul
    LogicalResult convertMulToEmitC(tosa::MulOp mulOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("mul_float");
        std::string varName = "t" + std::to_string(id++);                                            
        
        auto shiftAttr = mulOp->getAttrOfType<mlir::IntegerAttr>("shift");
        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1),
            rewriter.getIndexAttr(2),
            shiftAttr    
        });
        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(mulOp.getResult().getType())
        });
        
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
                mulOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                mulOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );

            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        convParamMap[callee.str()] = {};

        auto newCall = rewriter.create<emitc::CallOpaqueOp>(
            mulOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(),
            operands
        );
        rewriter.eraseOp(mulOp);
        return success();
    }

    //exp
    LogicalResult convertExpToEmitC(tosa::ExpOp expOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("exp");

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1)
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(expOp.getResult().getType())
        });

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
                expOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );

            auto addrOf = rewriter.create<emitc::ApplyOp>(
                expOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        convParamMap[callee.str()] = {};

        rewriter.create<emitc::CallOpaqueOp>(
            expOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );

        rewriter.eraseOp(expOp);
        return success();
    }

    //reciprocal
    LogicalResult convertReciprocalToEmitC(tosa::ReciprocalOp recipOp, PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("reciprocal");

        auto opaqueTensorType = emitc::OpaqueType::get(rewriter.getContext(), "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1)
        });

        ArrayAttr templateArgs = rewriter.getArrayAttr({
            TypeAttr::get(recipOp.getResult().getType())
        });

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
                recipOp.getLoc(),
                lvalueType,
                FlatSymbolRefAttr::get(rewriter.getContext(), symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                recipOp.getLoc(),
                ptrType,
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        convParamMap[callee.str()] = {};

        rewriter.create<emitc::CallOpaqueOp>(
            recipOp.getLoc(), 
            TypeRange{}, 
            callee, 
            args, 
            ArrayAttr(), 
            operands
        );

        rewriter.eraseOp(recipOp);
        return success();
    }

    //reshape
    LogicalResult convertReshapeToEmitC(tosa::ReshapeOp reshapeOp,
            PatternRewriter &rewriter) {
        StringAttr callee = rewriter.getStringAttr("reshape_s8");

        auto ctx = rewriter.getContext();
        auto opaqueTensorType = emitc::OpaqueType::get(ctx, "Tensor");
        auto lvalueType = emitc::LValueType::get(opaqueTensorType);
        auto ptrType = emitc::PointerType::get(opaqueTensorType);

        auto i32Type = rewriter.getI32Type();

        if (!zeroIndex || zeroIndex.getContext() != rewriter.getContext()) {
            zeroIndex = rewriter.create<emitc::ConstantOp>(
                rewriter.getUnknownLoc(),
                rewriter.getIndexType(),
                rewriter.getIndexAttr(0)
            );
        }
        ArrayAttr args = rewriter.getArrayAttr({
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(1)
        });
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
                reshapeOp.getLoc(), 
                lvalueType,
                FlatSymbolRefAttr::get(ctx, symbolName)
            );
            auto addrOf = rewriter.create<emitc::ApplyOp>(
                reshapeOp.getLoc(), 
                ptrType, 
                rewriter.getStringAttr("&"),
                getGlobal.getResult()
            );
            globalAddrMap[symbolName] = addrOf.getResult();
            operands.push_back(addrOf.getResult());
        }

        std::string shapeSymbol = symbolName + "_new_shape";
        auto shapeGlobal = rewriter.create<emitc::GetGlobalOp>(
            reshapeOp.getLoc(),
            emitc::ArrayType::get({(int64_t)reshapeOp.getNewShape().size()}, i32Type),
            FlatSymbolRefAttr::get(ctx, shapeSymbol)
        );

        auto arr0 = rewriter.create<emitc::SubscriptOp>(
            reshapeOp.getLoc(), 
            emitc::LValueType::get(i32Type), 
            shapeGlobal,
            SmallVector<Value, 1>{zeroIndex}
        );
        auto shapePtr = rewriter.create<emitc::ApplyOp>(
            reshapeOp.getLoc(), 
            emitc::PointerType::get(i32Type),
            rewriter.getStringAttr("&"), 
            arr0
        );

        operands.push_back(shapePtr.getResult());

        convParamMap[callee.str()] = {};
        rewriter.create<emitc::CallOpaqueOp>(
            reshapeOp.getLoc(),
            TypeRange{},        
            callee,
            ArrayAttr(), 
            ArrayAttr(),
            operands
        );

        rewriter.eraseOp(reshapeOp);
        return success();
    }

    struct TosaToEmitC
        : public mlir::PassWrapper<TosaToEmitC, OperationPass<ModuleOp>> {
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaToEmitC)

        TensorGraph* buildTensorGraph(mlir::ModuleOp module) {
            auto* graph = new TensorGraph();

            module.walk([&](mlir::Operation* op) {
                if (op->getName().getStringRef().starts_with("tosa.const")){
                    return;
                }
                auto* node = new TensorNode(op);
                graph->nodes.push_back(node);

                for (auto result : op->getResults()) {
                    graph->definingNodes[result] = node;
                }
                for (auto operand : op->getOperands()) {
                    graph->userNodes[operand].push_back(node);
                }
            });

            module.walk([&](mlir::func::FuncOp funcOp) {
                if (funcOp.getName() == "main") {
                    auto* mainNode = new TensorNode(funcOp);
                    graph->nodes.push_back(mainNode);

                    for (mlir::BlockArgument arg : funcOp.getArguments()) {
                        graph->definingNodes[arg] = mainNode;
                    }
                }
            });

            return graph;
        }


      void runOnOperation() override {

        mlir::ModuleOp module = getOperation();
        mlir::OpBuilder builder(module.getContext());
        mlir::MLIRContext *context = module.getContext();
        
        
        llvm::SmallVector<std::tuple<std::string, mlir::ElementsAttr, mlir::RankedTensorType, 
                    std::optional<mlir::DenseElementsAttr>, std::optional<mlir::DenseElementsAttr> >, 8> globalVars;
        llvm::SmallVector<std::tuple<std::string, int, mlir::RankedTensorType,
                    std::optional<mlir::DenseElementsAttr>, std::optional<mlir::DenseElementsAttr> >, 8> notConstGlobalVars;
        llvm::SmallVector<std::tuple<std::string, int, mlir::RankedTensorType>, 4> inputArgs;

        builder.setInsertionPointToStart(module.getBody());

        builder.create<emitc::IncludeOp>(
            module.getLoc(),
            builder.getStringAttr("reference.h")
        );

        builder.create<emitc::IncludeOp>(
            module.getLoc(),
            builder.getStringAttr("cmsis-nn.h")
        );

        builder.create<emitc::VerbatimOp>(
            module.getLoc(),
            builder.getStringAttr("extern \"C\" {")
        );
        static int argIdx = 0;

        TensorGraph *graph = buildTensorGraph(module);
        LivenessAnalysis *liveness = new LivenessAnalysis(graph);
        MemoryPlanner memoryPlanner(graph, liveness);
        memoryPlanner.computeTensorSizes();
        memoryPlanner.performMemoryOptimizer();

        module.walk([&](mlir::func::FuncOp funcOp){
            for(unsigned i = 0; i<funcOp.getNumArguments(); ++i){
                mlir::BlockArgument arg = funcOp.getArgument(i);

                std::string varName = "t_arg"+std::to_string(argIdx++);
                
                auto argType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
                if(!argType) { continue; }

                auto opaqueTensorType = emitc::OpaqueType::get(context, "Tensor");
                auto lvalueType = emitc::LValueType::get(opaqueTensorType);

                auto globalTensor = builder.create<emitc::GlobalOp>(
                        funcOp.getLoc(),
                        builder.getStringAttr(varName),
                        mlir::TypeAttr::get(opaqueTensorType),
                        Attribute(),
                        nullptr,
                        nullptr,
                        nullptr
                        );
                globalMap[arg.getAsOpaquePointer()] = globalTensor;

                //shape
                std::string shapeVarName = varName + "_shape";
                auto shape = argType.getShape();
                SmallVector<Attribute> shapeAttrValues;
                for (int64_t dim : shape) {
                    shapeAttrValues.push_back(builder.getI32IntegerAttr(dim));
                }

                auto shapeTensorType = RankedTensorType::get({(int64_t)shape.size()}, builder.getI32Type());
                auto shapeAttr = DenseElementsAttr::get(shapeTensorType, shapeAttrValues);
                auto shapeType = emitc::ArrayType::get({(int64_t)shape.size()}, builder.getI32Type());

                builder.create<emitc::GlobalOp>(
                        builder.getUnknownLoc(),
                        builder.getStringAttr(shapeVarName),
                        mlir::TypeAttr::get(shapeType),
                        shapeAttr,
                        nullptr,
                        builder.getUnitAttr(),
                        nullptr //builder.getUnitAttr()
                        );
                std::string rankVarName = varName + "_rank";
                int64_t rank = shape.size();
                auto rankAttr = builder.getI32IntegerAttr(rank);

                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(rankVarName),
                    mlir::TypeAttr::get(builder.getI32Type()),
                    rankAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );
                //location - offset
                size_t offset = 0;
                const AllocationInfo* info = memoryPlanner.getAllocationInfo(arg);
                if (info) {
                    offset = info->offset;
                }
                inputArgs.push_back(std::make_tuple(varName, offset, argType));

            }
        });

        static int tensorIdx = 0;
        
        module.walk([&](mlir::Operation *op) {

            std::string varName = "t" + std::to_string(tensorIdx++);
          
            if (op->getNumResults() == 0){return;}

            auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
            if (!resultType){
                if (auto constOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
                    return ;
                }
            }

            auto opaqueTensorType = emitc::OpaqueType::get(builder.getContext(), "Tensor");
            auto lvalueType = emitc::LValueType::get(opaqueTensorType);

            auto globalTensor = builder.create<emitc::GlobalOp>(
                builder.getUnknownLoc(),
                builder.getStringAttr(varName),
                mlir::TypeAttr::get(opaqueTensorType),
                Attribute(),
                nullptr,
                builder.getUnitAttr(),
                nullptr
            );

            globalMap[op->getResult(0).getAsOpaquePointer()] = globalTensor;
            
            std::optional<DenseElementsAttr> scaleAttrOpt = std::nullopt;
            std::optional<DenseElementsAttr> zeroPointAttrOpt = std::nullopt;
          
            //constOp
            if(auto tosaConstOp = llvm::dyn_cast<tosa::ConstOp>(op)){
                auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tosaConstOp.getType());

                if (!tensorType){return;}
            
                auto shape = tensorType.getShape();
                auto elemType = tensorType.getElementType();

                bool isQuant = false;
                float scale = 1.0f;
                float zeroPoint = 0.0f;
                Type shapeElemType = builder.getIntegerType(32);
                auto shapeArrayType = emitc::ArrayType::get({(int64_t)shape.size()}, shapeElemType);
                
                //shape
                SmallVector<Attribute> shapeValues;
                for (int64_t dim : tensorType.getShape()){
                    shapeValues.push_back(builder.getI32IntegerAttr(dim));
                }
                auto shapeType = emitc::ArrayType::get({(int64_t)shapeValues.size()}, builder.getI32Type());
                auto shapeAttr = DenseElementsAttr::get(
                    RankedTensorType::get({(int64_t)shapeValues.size()},
                    builder.getI32Type()),
                    shapeValues
                );

                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(varName + "_shape"),
                    TypeAttr::get(shapeType),
                    shapeAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );
                std::string rankVarName = varName + "_rank";
                int64_t rank = shape.size();
                auto rankAttr = builder.getI32IntegerAttr(rank);
                
                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(rankVarName),
                    mlir::TypeAttr::get(builder.getI32Type()),
                    rankAttr,     
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );

                //value
                int64_t elementCount = tensorType.getNumElements();
                auto valueElementType = tensorType.getElementType();
                auto valueAttr = tosaConstOp.getValue();
                if (!valueAttr){return ; }
                
                if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(valueElementType)) {
                    valueElementType = quantType.getStorageType();
                }
                else if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(valueElementType)) {
                    valueElementType = perAxisType.getStorageType();
                }
                SmallVector<int64_t, 1> flatShape = {elementCount};


                auto flatArrayType = emitc::ArrayType::get(flatShape, valueElementType);
                auto rawTensorType = RankedTensorType::get(flatShape, valueElementType);
                
                auto denseAttr = llvm::cast<DenseElementsAttr>(valueAttr);
                SmallVector<Attribute> valueList;
                for (Attribute val : denseAttr.getValues<Attribute>()){
                    valueList.push_back(val);
                }
                auto flatValueAttr = DenseElementsAttr::get(rawTensorType, valueList);

                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(varName + "_data"),
                    TypeAttr::get(flatArrayType),
                    flatValueAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );

                globalVars.push_back(std::make_tuple(varName, valueAttr, tensorType, scaleAttrOpt, zeroPointAttrOpt));
            }
            else if (auto constShapeOp = llvm::dyn_cast<tosa::ConstShapeOp>(op)) {
                auto valueAttr = constShapeOp.getValue();
                auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr);
                if (!denseAttr)
                    return;

                auto elementType = denseAttr.getElementType();
                if (!mlir::isa<mlir::IntegerType>(elementType) && !elementType.isIndex()) {
                    llvm::errs() << "Unsupported element type in tosa.const_shape\n";
                return;
                }

                auto i32Type = builder.getI32Type();

                SmallVector<int32_t> dataValues;
                for (APInt val : denseAttr.getValues<APInt>()) {
                    dataValues.push_back(static_cast<int32_t>(val.getSExtValue()));
                }

                auto dataShape = llvm::ArrayRef<int64_t>{static_cast<int64_t>(dataValues.size())};
                auto dataArrayType = emitc::ArrayType::get(dataShape, i32Type);
                auto dataTensorType = RankedTensorType::get(dataShape, i32Type);
                auto dataAttr = DenseElementsAttr::get<int32_t>(dataTensorType, dataValues);

                int32_t shapeLength = static_cast<int32_t>(dataValues.size());
                auto shapeArrayType = emitc::ArrayType::get({1}, i32Type);
                auto shapeTensorType = RankedTensorType::get({1}, i32Type);
                auto shapeAttr = DenseElementsAttr::get<int32_t>(shapeTensorType, {shapeLength});

                auto shapeGlobal = builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(varName + "_shape"),
                    TypeAttr::get(shapeArrayType),
                    shapeAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );
                llvm::SmallVector<int64_t, 4> shape;
                int64_t rank = 0;
                mlir::IntegerAttr rankAttr;

                auto resultType = constShapeOp.getResult().getType();

                if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
                    shape = llvm::to_vector<4>(tensorType.getShape());
                    rank = static_cast<int64_t>(shape.size());
                    rankAttr = builder.getI32IntegerAttr(rank);
                }else {
                    auto valueAttr = constShapeOp.getValue();
                    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
                        rank = static_cast<int64_t>(denseAttr.getNumElements());
                        shape.push_back(rank);
                        rankAttr = builder.getI32IntegerAttr(rank);
                    } else {
                        llvm::errs() << "Error: const_shape has invalid value\n";
                        return;
                    }
                }

                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(varName + "_rank"),
                    mlir::TypeAttr::get(builder.getI32Type()),
                    rankAttr,  
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );

                auto dataGlobal = builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(varName + "_data"),
                    TypeAttr::get(dataArrayType),
                    dataAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                );

                globalVars.push_back(std::make_tuple(varName, valueAttr, dataTensorType, scaleAttrOpt, zeroPointAttrOpt));
            } 
            // Not constOp
            else{
                if (op->getNumResults() == 0){ return; }
                auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
                if (!resultType) {return ;}

                auto shape = resultType.getShape();
                auto elemType = resultType.getElementType();

                bool isQuant = false;
                float scale = 1.0f;
                float zeroPoint = 0.0f;
                Type shapeElemType = builder.getI32Type();
                
                if(auto quantType= mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)){
                    isQuant = true;
                    shapeElemType = quantType.getExpressedType();
                    scale = static_cast<float>(quantType.getScale());
                    zeroPoint = static_cast<float>(quantType.getZeroPoint());
                }
            
                SmallVector<mlir::Attribute, 4> shapeValues;
                for (int64_t d : shape){ shapeValues.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(d))); }
                
                auto shapeArrayType = emitc::ArrayType::get({(int64_t)shapeValues.size()}, builder.getI32Type());

                auto shapeAttr = DenseElementsAttr::get(
                    RankedTensorType::get({(int64_t)shapeValues.size()},
                    builder.getI32Type()),
                    shapeValues
                );

                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr( varName + "_shape"),
                    TypeAttr::get(shapeArrayType),
                    shapeAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr 
                );
                std::string rankVarName = varName + "_rank";
                int64_t rank = shape.size();
                auto rankAttr = builder.getI32IntegerAttr(rank);
                
                builder.create<emitc::GlobalOp>(
                    builder.getUnknownLoc(),
                    builder.getStringAttr(rankVarName),
                    mlir::TypeAttr::get(builder.getI32Type()),
                    rankAttr,
                    nullptr,
                    builder.getUnitAttr(),
                    nullptr
                ); 
                //rescale Multiplier & Shift
                if (auto rescaleOp = llvm::dyn_cast<tosa::RescaleOp>(op)) {
                    auto multiplierAttr = rescaleOp.getMultiplierAttr();
                    auto shiftAttr = rescaleOp.getShiftAttr();
                    std::vector<int32_t> multiplierVec;
                    for (int32_t v : multiplierAttr.asArrayRef()) {
                        multiplierVec.push_back(v);
                    }
                    std::vector<int32_t> shiftVec;
                    for(int32_t v : shiftAttr.asArrayRef()) {
                        shiftVec.push_back(v);
                    }
                    auto arrayTypeM = emitc::ArrayType::get({(int64_t)multiplierVec.size()}, builder.getI32Type());
                    auto dataAttrM = DenseElementsAttr::get(
                        RankedTensorType::get({(int64_t)multiplierVec.size()}, builder.getI32Type()),
                        llvm::ArrayRef<int32_t>(multiplierVec)
                    );

                    builder.create<emitc::GlobalOp>(
                        builder.getUnknownLoc(),
                        builder.getStringAttr(varName + "_multiplier"),
                        TypeAttr::get(arrayTypeM),
                        dataAttrM,
                        nullptr,
                        builder.getUnitAttr(),
                        nullptr
                    );

                    auto arrayTypeS = emitc::ArrayType::get({(int64_t)shiftVec.size()}, builder.getI32Type());
                    auto dataAttrS = DenseElementsAttr::get(
                        RankedTensorType::get({(int64_t)shiftVec.size()}, builder.getI32Type()),
                        llvm::ArrayRef<int32_t>(shiftVec)
                    );

                    builder.create<emitc::GlobalOp>(
                        builder.getUnknownLoc(),
                        builder.getStringAttr(varName + "_shift"),
                        TypeAttr::get(arrayTypeS),
                        dataAttrS,
                        nullptr,
                        builder.getUnitAttr(),
                        nullptr
                    );
                }

                if (auto reshapeOp = llvm::dyn_cast<tosa::ReshapeOp>(op)) {
                    auto newShape = reshapeOp.getNewShape();
            
                    std::vector<int32_t> shapeVec;
                    shapeVec.reserve(newShape.size());
                    for (int64_t dim : newShape) {
                        shapeVec.push_back(static_cast<int32_t>(dim));
                    }

                    auto arrayType = emitc::ArrayType::get({(int64_t)shapeVec.size()}, builder.getI32Type());
                
                    auto dataAttr = DenseElementsAttr::get(
                        RankedTensorType::get({(int64_t)shapeVec.size()}, builder.getI32Type()),
                        llvm::ArrayRef<int32_t>(shapeVec)
                    );

                    builder.create<emitc::GlobalOp>(
                        builder.getUnknownLoc(),
                        builder.getStringAttr(varName + "_new_shape"),
                        TypeAttr::get(arrayType),
                        dataAttr,
                        nullptr,
                        builder.getUnitAttr(),  
                        nullptr
                    );
                }

                mlir::Value result = op->getResult(0); 
                
                size_t offsetResult = 0;
                const AllocationInfo* info = memoryPlanner.getAllocationInfo(result);
                if (info) {
                    offsetResult = info->offset;
                } 
                notConstGlobalVars.push_back(std::make_tuple(varName, offsetResult, resultType, scaleAttrOpt, zeroPointAttrOpt));
            }

        });

        int64_t arenaSize = static_cast<int64_t>(memoryPlanner.getPeakMemoryUsage());
        auto arenaArrayType = emitc::ArrayType::get({arenaSize}, builder.getI8Type());
        builder.create<emitc::GlobalOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr("arena"),   
            TypeAttr::get(arenaArrayType),    
            nullptr,                          
            nullptr,                             
            builder.getUnitAttr()       
        );
      
        func::FuncOp modelInitFunc = module.lookupSymbol<func::FuncOp>("model_init");
        if (!modelInitFunc) {
            FunctionType funcType = builder.getFunctionType({}, {});
            modelInitFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(), "model_init", funcType);
            Block *entryBlock = modelInitFunc.addEntryBlock();
            builder.setInsertionPointToEnd(entryBlock);
        }else {
            Block *entryBlock = &modelInitFunc.getBody().front();
            builder.setInsertionPointToEnd(entryBlock);
        }
        
        size_t currentOffset = 0;
        auto zeroIndex1 = builder.create<emitc::ConstantOp>(
            builder.getUnknownLoc(),
            builder.getIndexType(),
            builder.getIndexAttr(0)
        );

        auto arenaGlobal = builder.create<emitc::GetGlobalOp>(
            builder.getUnknownLoc(),
            arenaArrayType,
            builder.getStringAttr("arena")
        );
        auto arenaElemLValue = builder.create<emitc::SubscriptOp>(
            builder.getUnknownLoc(),
            emitc::LValueType::get(builder.getI8Type()),
            arenaGlobal,
            SmallVector<Value, 1>{zeroIndex1}
        );
        auto arenaPtr = builder.create<emitc::ApplyOp>(
            builder.getUnknownLoc(),
            emitc::PointerType::get(builder.getI8Type()),
            builder.getStringAttr("&"),
            arenaElemLValue
        );

        //init cosnt
        for (auto &[varName, varValue, tensorType, scaleAttrOpt, zeroPointAttrOpt] : globalVars) {
            //Tensor getGlobal
            auto tensorOpaqueType = emitc::OpaqueType::get(builder.getContext(), "Tensor");
            auto lvalueTensorType = emitc::LValueType::get(tensorOpaqueType);

            auto getGlobal = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                lvalueTensorType,
                FlatSymbolRefAttr::get(builder.getContext(), varName)
            );

            //shape 
            auto shapePtrType = emitc::PointerType::get(builder.getI32Type());
            auto shapeLValueType = emitc::LValueType::get(shapePtrType);

            auto tVarShapeMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                shapeLValueType,
                builder.getStringAttr("shape"),
                getGlobal
            );
            auto shapeArrayType = emitc::ArrayType::get({(int64_t)tensorType.getRank()}, builder.getI32Type());
            auto getGlobalShape = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                shapeArrayType,
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_shape")
            );
            auto arr0 = builder.create<emitc::SubscriptOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                getGlobalShape,
                SmallVector<Value, 1>{zeroIndex1}
            );
            auto addr = builder.create<emitc::ApplyOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(builder.getI32Type()),
                builder.getStringAttr("&"),
                arr0
            );
            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarShapeMember,
                addr
            );

            // rank
            auto tVarRankMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                builder.getStringAttr("rank"),
                getGlobal 
            );

            auto getGlobalRank = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                FlatSymbolRefAttr::get(builder.getContext(),varName + "_rank")
            );
            auto loadRank = builder.create<emitc::LoadOp>(
                builder.getUnknownLoc(),
                builder.getI32Type(),
                getGlobalRank
            );

            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarRankMember,
                loadRank
            );

            //data (location)
            auto voidPtrType = emitc::PointerType::get(builder.getI8Type());
            auto locationLValueType = emitc::LValueType::get(voidPtrType);
            auto elemType = tensorType.getElementType();
            if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
                elemType = quantType.getStorageType();
            } else if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
                elemType = perAxisType.getStorageType();
            }
            auto dataArrayType = emitc::ArrayType::get({(int64_t)tensorType.getNumElements()}, elemType);
            auto tVarLocationMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                locationLValueType,
                builder.getStringAttr("location"),
                getGlobal
            );
            auto getGlobalData = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                dataArrayType,
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_data")
            );
            auto dataArr0 = builder.create<emitc::SubscriptOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(elemType),
                getGlobalData,
                SmallVector<Value, 1>{zeroIndex1}
            );
            auto dataAddr = builder.create<emitc::ApplyOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(elemType), 
                builder.getStringAttr("&"),
                dataArr0
            );
            auto castDataAddr = builder.create<emitc::CastOp>(
                builder.getUnknownLoc(),
                voidPtrType,
                dataAddr
            );
            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarLocationMember,
                castDataAddr
            );
        
            auto floatType = builder.getF32Type();
            auto floatPtrType = emitc::PointerType::get(floatType);
            auto floatLValueType = emitc::LValueType::get(floatType);
            auto floatLValuePtrType = emitc::LValueType::get(floatPtrType);
            
            auto int32Type = builder.getI32Type();
            auto int32PtrType = emitc::PointerType::get(int32Type);
            auto int32LValuePtrType = emitc::LValueType::get(int32PtrType);

        }
        //init operation tensor
        for (auto &[varName, offset, tensorType, scaleAttrOpt, zeroPointAttrOpt] : notConstGlobalVars) {
            
            auto floatType = builder.getF32Type();
            auto floatPtrType = emitc::PointerType::get(floatType);
            auto floatLValueType = emitc::LValueType::get(floatType);
            auto floatLValuePtrType = emitc::LValueType::get(floatPtrType);

            auto int32Type = builder.getI32Type();
            auto int32PtrType = emitc::PointerType::get(int32Type);
            auto int32LValuePtrType = emitc::LValueType::get(int32PtrType);


            auto tensorOpaqueType = emitc::OpaqueType::get(builder.getContext(), "Tensor");
            auto lvalueTensorType = emitc::LValueType::get(tensorOpaqueType);

            auto getGlobal = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                lvalueTensorType,
                FlatSymbolRefAttr::get(builder.getContext(), varName)
            );
            //shape
            auto shapeArrayType = emitc::ArrayType::get({(int64_t)tensorType.getRank()}, builder.getI32Type());
            auto shapePtrType = emitc::PointerType::get(builder.getI32Type());
            auto shapeLValueType = emitc::LValueType::get(shapePtrType);

            auto tVarShapeMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                shapeLValueType,
                builder.getStringAttr("shape"),
                getGlobal
            );
            auto getGlobalShape = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                shapeArrayType,
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_shape")
            );
            auto arr0 = builder.create<emitc::SubscriptOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                getGlobalShape,
                SmallVector<Value, 1>{zeroIndex1}
            );
            auto addr = builder.create<emitc::ApplyOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(builder.getI32Type()),
                builder.getStringAttr("&"),
                arr0
            );
            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarShapeMember,
                addr
            );
            // rank 
            auto tVarRankMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                builder.getStringAttr("rank"),
                getGlobal
            );

            auto getGlobalRank = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                //builder.getI32Type(),
                emitc::LValueType::get(builder.getI32Type()),
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_rank")
            );
            auto loadRank = builder.create<emitc::LoadOp>(
                builder.getUnknownLoc(),
                builder.getI32Type(),
                getGlobalRank
            );

            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarRankMember,
                loadRank
            ); 
            //location
            auto voidPtrType = emitc::PointerType::get(builder.getI8Type());
            auto locationLValueType = emitc::LValueType::get(voidPtrType);

            auto tVarLocationMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                locationLValueType,
                builder.getStringAttr("location"),
                getGlobal
            );
            auto offsetAttr = builder.getI64IntegerAttr(offset);

            auto offsetConst = builder.create<emitc::ConstantOp>(
                builder.getUnknownLoc(),
                builder.getIntegerType(64),
                offsetAttr
            );
            auto arenaLocation = builder.create<emitc::AddOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(builder.getI8Type()),
                arenaPtr,
                offsetConst
            );

            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarLocationMember,
                arenaLocation
            );
        }

        for (auto &[varName, offset, tensorType] : inputArgs) {

            auto tensorOpaqueType = emitc::OpaqueType::get(builder.getContext(), "Tensor");
            auto lvalueTensorType = emitc::LValueType::get(tensorOpaqueType);

            auto getGlobal = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                lvalueTensorType,
                FlatSymbolRefAttr::get(builder.getContext(), varName)
            );
            //shape
            auto shapeArrayType = emitc::ArrayType::get({(int64_t)tensorType.getRank()}, builder.getI32Type());
            auto shapePtrType = emitc::PointerType::get(builder.getI32Type());
            auto shapeLValueType = emitc::LValueType::get(shapePtrType);

            auto tVarShapeMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                shapeLValueType,
                builder.getStringAttr("shape"),
                getGlobal
            );
            auto getGlobalShape = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                shapeArrayType,
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_shape")
            );
            auto arr0 = builder.create<emitc::SubscriptOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                getGlobalShape,
                SmallVector<Value, 1>{zeroIndex1}
            );
            auto addr = builder.create<emitc::ApplyOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(builder.getI32Type()),
                builder.getStringAttr("&"),
                arr0
            );
            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarShapeMember,
                addr
            );

            // rank
            auto tVarRankMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                builder.getStringAttr("rank"),
                getGlobal 
            );

            auto getGlobalRank = builder.create<emitc::GetGlobalOp>(
                builder.getUnknownLoc(),
                emitc::LValueType::get(builder.getI32Type()),
                FlatSymbolRefAttr::get(builder.getContext(), varName + "_rank")
            );
            auto loadRank = builder.create<emitc::LoadOp>(
                builder.getUnknownLoc(),
                builder.getI32Type(),
                getGlobalRank
            );

            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarRankMember,
                loadRank
            );

            //location
            auto voidPtrType = emitc::PointerType::get(builder.getI8Type());
            auto locationLValueType = emitc::LValueType::get(voidPtrType);

            auto tVarLocationMember = builder.create<emitc::MemberOp>(
                builder.getUnknownLoc(),
                locationLValueType,
                builder.getStringAttr("location"),
                getGlobal
            );
            auto offsetAttr = builder.getI64IntegerAttr(offset);

            auto offsetConst = builder.create<emitc::ConstantOp>(
                builder.getUnknownLoc(),
                builder.getIntegerType(64),
                offsetAttr
            );
            
            auto arenaLocation = builder.create<emitc::AddOp>(
                builder.getUnknownLoc(),
                emitc::PointerType::get(builder.getI8Type()),
                arenaPtr,
                offsetConst
            );

            builder.create<emitc::AssignOp>(
                builder.getUnknownLoc(),
                tVarLocationMember,
                arenaLocation
            );
        }

        if (modelInitFunc.getBody().front().empty() ||
                !isa<func::ReturnOp>(modelInitFunc.getBody().front().back())) {
            builder.create<func::ReturnOp>(builder.getUnknownLoc());
        }
        
        //main
        func::FuncOp mainFunc = module.lookupSymbol<func::FuncOp>("main");
        auto retTy = emitc::PointerType::get(emitc::OpaqueType::get(context, "Tensor"));
        mainFunc.setFunctionType(FunctionType::get(
            context,
            mainFunc.getFunctionType().getInputs(),
            {retTy}
        ));
        
        mainFunc.setName("model");
        
        Block *entryBlock = &mainFunc.getBody().front();
        builder.setInsertionPointToStart(entryBlock);

        mainFunc.walk([&](func::ReturnOp retOp) {
            mlir::Value retVal = retOp.getOperand(0);
            //remove cast
            if(auto castOp = retVal.getDefiningOp<mlir::UnrealizedConversionCastOp>()){
                if(castOp.getOperands().size() == 1){
                    retVal = castOp.getOperands()[0];
                    castOp.erase();
                }
            }
            auto it = globalMap.find(retVal.getAsOpaquePointer());
            if(it == globalMap.end()){
                retOp.emitError("Missing global mapping for return value");
                return ;
            }
            std::string symbolName = it -> second.getSymName().str();
            auto opaqueTensorType = emitc::OpaqueType::get(context, "Tensor");
            auto lvalueType =  emitc::LValueType::get(opaqueTensorType);
            auto ptrType = emitc::PointerType::get(opaqueTensorType);

            builder.setInsertionPoint(retOp);
            auto getGlobal = builder.create<emitc::GetGlobalOp>(
                retOp.getLoc(), 
                lvalueType,
                FlatSymbolRefAttr::get(context, symbolName)
            );
            auto addrOf = builder.create<emitc::ApplyOp>(
                retOp.getLoc(),
                ptrType,
                builder.getStringAttr("&"),
                getGlobal
            );
            builder.create<mlir::func::ReturnOp>(retOp.getLoc(), addrOf.getResult());
            retOp.erase();
        });
        builder.setInsertionPointToEnd(module.getBody());
        builder.create<emitc::VerbatimOp>(
            module.getLoc(),
            builder.getStringAttr("}")
        );
        llvm::errs()<<"\n";
        // Define the conversion target
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<emitc::EmitCDialect, arith::ArithDialect,
                              func::FuncDialect, memref::MemRefDialect>();
        target.addLegalOp<mlir::ModuleOp>();
        target.addIllegalDialect<tosa::TosaDialect, tensor::TensorDialect>();

        // Define the rewrite patterns
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add(convertMax_pool2DToEmitC);
        patterns.add(convertAddToEmitC);
        patterns.add(convertSubToEmitC);
        patterns.add(convertMulToEmitC);
        patterns.add(convertRescaleToEmitC);
        patterns.add(convertTransposeConv2DToEmitC);
        patterns.add(convertPadToEmitC);
        patterns.add(convertConcatToEmitC);
        patterns.add(convertCastToEmitC);
        patterns.add(convertConstToEmitC);
        patterns.add(convertConv2DToEmitC);
        patterns.add(convertConstShapeToEmitC);
        patterns.add(convertDepthwiseConv2DToEmitC);
        patterns.add(convertReduceSumToEmitC);
        patterns.add(convertReduceMaxToEmitC);
        patterns.add(convertReshapeToEmitC);
        patterns.add(convertExpToEmitC);
        patterns.add(convertReciprocalToEmitC);
        patterns.add(convertAvg_pool2DToEmitC);
        patterns.add(convertTransposeToEmitC);

        // Apply the full conversion
        if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns)))){
          return  signalPassFailure();
        }
        
        //remove main argument
        mainFunc = module.lookupSymbol<func::FuncOp>("model");
        if (!mainFunc) {
            module.emitError("main function not found after conversion");
            return signalPassFailure();
        }

        auto funcType = mainFunc.getFunctionType();
        int64_t numArgs = funcType.getNumInputs();

        if (numArgs > 0) {
            Block *entryBlock = &mainFunc.getBody().front();

            for (int64_t i = numArgs - 1; i >= 0; --i) {
                mlir::BlockArgument arg = mainFunc.getArgument(i);
                    
                if (arg.use_empty()) {
                    mainFunc.eraseArgument(i);                                                                        }                                                                                                     
                else{                                                                                                   arg.getParentBlock()->getParentOp()->emitError()
                    << "Cannot erase argument " << i << " because it is still in use.";
                }

            }

            FunctionType newType = FunctionType::get(context, {}, {retTy});
            mainFunc.setFunctionType(newType);
            mainFunc.setAllArgAttrs(mlir::ArrayAttr::get(context, {}));
        }
        
        std::ofstream fout2("gen/model_ops_list.txt");
        for (const auto &kv : convParamMap) {
            fout2 << kv.first << " "; 
            for (auto v : kv.second) { 
                fout2 << v << " ";
            }
            fout2 << "\n";
        }
        fout2.close();
      }
    };
}
std::unique_ptr<mlir::Pass> mlir::tosa::createTosaToEmitC() {
  return std::make_unique<TosaToEmitC>();
}

