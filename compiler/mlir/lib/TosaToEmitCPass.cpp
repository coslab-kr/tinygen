#include "MemoryPlanner/LivenessAnalysis.h"
#include "MemoryPlanner/MemoryPlanner.h"
#include "Passes.h"
#include "TosaToEmitC/Lowering.h"
#include "TosaToEmitC/TensorEmitter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;
using namespace tinygen;

namespace {
template <typename OpT>
class LoweringPattern final : public OpRewritePattern<OpT> {
public:
  using Callback = std::function<LogicalResult(OpT, PatternRewriter &)>;
  LoweringPattern(MLIRContext *context, Callback callback)
      : OpRewritePattern<OpT>(context), callback(std::move(callback)) {}
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    return callback(op, rewriter);
  }

private:
  Callback callback;
};

template <typename OpT>
void addPattern(
    RewritePatternSet &patterns, TosaToEmitCLowering &lowering,
    LogicalResult (TosaToEmitCLowering::*convert)(OpT, PatternRewriter &)) {
  patterns.add<LoweringPattern<OpT>>(
      patterns.getContext(),
      [&lowering, convert](OpT op, PatternRewriter &rewriter) {
        return (lowering.*convert)(op, rewriter);
      });
}
} // namespace

std::unique_ptr<TensorGraph>
TosaToEmitCLowering::buildTensorGraph(func::FuncOp main) {
  auto graph = std::make_unique<TensorGraph>();
  auto *inputs = graph->addNode(main);
  inputs->inputs.clear();
  for (Value arg : main.getArguments()) {
    inputs->outputs.push_back(arg);
  }
  // Lifetimes follow the emitted straight-line program's execution order.
  for (Operation &op : main.front()) {
    if (isa<tosa::ConstOp, tosa::ConstShapeOp>(op))
      continue;
    graph->addNode(&op);
  }
  return graph;
}

LogicalResult TosaToEmitCLowering::run(ModuleOp module,
                                       const std::string &opsFile) {

  auto entry = module.lookupSymbol<func::FuncOp>("main");
  if (!entry || entry.isDeclaration() ||
      !llvm::hasSingleElement(entry.getBody()))
    return module.emitError("expected one defined, single-block main function");
  if (std::distance(module.getOps<func::FuncOp>().begin(),
                    module.getOps<func::FuncOp>().end()) != 1)
    return module.emitError("only one function is supported");
  auto ret = dyn_cast<func::ReturnOp>(entry.front().getTerminator());
  if (!ret || ret.getNumOperands() != 1)
    return entry.emitError("main must return exactly one tensor");
  auto validTensor = [](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape() && tensor.getRank() > 0 &&
           llvm::all_of(tensor.getShape(), [](int64_t dim) {
             return dim > 0 && dim <= std::numeric_limits<int32_t>::max();
           });
  };
  for (Value arg : entry.getArguments())
    if (!validTensor(arg.getType()))
      return entry.emitError(
          "main arguments must be nonempty, static ranked tensors");
  if (!validTensor(ret.getOperand(0).getType()))
    return entry.emitError("main must return a nonempty, static ranked tensor");
  for (Operation &op : entry.front()) {
    if (op.getNumRegions() || op.getNumResults() > 1)
      return op.emitError(
          "nested regions and multiple results are not supported");
    for (Type type : op.getResultTypes())
      if (!validTensor(type) && !isa<tosa::ConstShapeOp>(op))
        return op.emitError("expected a nonempty, static ranked tensor result");
  }

  mlir::OpBuilder builder(module.getContext());
  mlir::MLIRContext *context = module.getContext();

  SmallVector<std::pair<std::string, RankedTensorType>, 8> globalVars;
  SmallVector<std::tuple<std::string, size_t, RankedTensorType>, 8>
      notConstGlobalVars;
  SmallVector<std::tuple<std::string, size_t, RankedTensorType>, 4> inputArgs;

  builder.setInsertionPointToStart(module.getBody());

  builder.create<emitc::IncludeOp>(module.getLoc(),
                                   builder.getStringAttr("reference.h"));

  builder.create<emitc::IncludeOp>(module.getLoc(),
                                   builder.getStringAttr("cmsis-nn.h"));

  builder.create<emitc::VerbatimOp>(module.getLoc(),
                                    builder.getStringAttr("extern \"C\" {"));
  int argIdx = 0;

  auto graph = buildTensorGraph(entry);
  LivenessAnalysis liveness(graph.get());
  MemoryPlanner memoryPlanner(graph.get(), &liveness);
  if (failed(memoryPlanner.computeTensorSizes()) ||
      failed(memoryPlanner.performMemoryOptimizer()))
    return failure();

  module.walk([&](mlir::func::FuncOp funcOp) {
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      mlir::BlockArgument arg = funcOp.getArgument(i);

      std::string varName = "t_arg" + std::to_string(argIdx++);

      auto argType = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
      if (!argType) {
        continue;
      }

      auto opaqueTensorType = emitc::OpaqueType::get(context, "Tensor");
      auto lvalueType = emitc::LValueType::get(opaqueTensorType);

      auto globalTensor = builder.create<emitc::GlobalOp>(
          funcOp.getLoc(), builder.getStringAttr(varName),
          mlir::TypeAttr::get(opaqueTensorType), Attribute(), nullptr, nullptr,
          nullptr);
      globalMap[arg.getAsOpaquePointer()] = globalTensor;

      // shape
      std::string shapeVarName = varName + "_shape";
      auto shape = argType.getShape();
      SmallVector<Attribute> shapeAttrValues;
      for (int64_t dim : shape) {
        shapeAttrValues.push_back(builder.getI32IntegerAttr(dim));
      }

      auto shapeTensorType =
          RankedTensorType::get({(int64_t)shape.size()}, builder.getI32Type());
      auto shapeAttr = DenseElementsAttr::get(shapeTensorType, shapeAttrValues);
      auto shapeType =
          emitc::ArrayType::get({(int64_t)shape.size()}, builder.getI32Type());

      builder.create<emitc::GlobalOp>(builder.getUnknownLoc(),
                                      builder.getStringAttr(shapeVarName),
                                      mlir::TypeAttr::get(shapeType), shapeAttr,
                                      nullptr, builder.getUnitAttr(),
                                      nullptr // builder.getUnitAttr()
      );
      std::string rankVarName = varName + "_rank";
      int64_t rank = shape.size();
      auto rankAttr = builder.getI32IntegerAttr(rank);

      builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(rankVarName),
          mlir::TypeAttr::get(builder.getI32Type()), rankAttr, nullptr,
          builder.getUnitAttr(), nullptr);
      // location - offset
      size_t offset = 0;
      const AllocationInfo *info = memoryPlanner.getAllocationInfo(arg);
      if (info) {
        offset = info->offset;
      }
      inputArgs.push_back(std::make_tuple(varName, offset, argType));
    }
  });

  int tensorIdx = 0;

  module.walk([&](mlir::Operation *op) {
    std::string varName = "t" + std::to_string(tensorIdx++);

    if (op->getNumResults() == 0) {
      return;
    }

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      if (auto constOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
        return;
      }
    }

    auto opaqueTensorType =
        emitc::OpaqueType::get(builder.getContext(), "Tensor");
    auto lvalueType = emitc::LValueType::get(opaqueTensorType);

    auto globalTensor = builder.create<emitc::GlobalOp>(
        builder.getUnknownLoc(), builder.getStringAttr(varName),
        mlir::TypeAttr::get(opaqueTensorType), Attribute(), nullptr,
        builder.getUnitAttr(), nullptr);

    globalMap[op->getResult(0).getAsOpaquePointer()] = globalTensor;

    // constOp
    if (auto tosaConstOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(tosaConstOp.getType());

      if (!tensorType) {
        return;
      }

      auto shape = tensorType.getShape();
      auto elemType = tensorType.getElementType();

      bool isQuant = false;
      float scale = 1.0f;
      float zeroPoint = 0.0f;
      Type shapeElemType = builder.getIntegerType(32);
      auto shapeArrayType =
          emitc::ArrayType::get({(int64_t)shape.size()}, shapeElemType);

      // shape
      SmallVector<Attribute> shapeValues;
      for (int64_t dim : tensorType.getShape()) {
        shapeValues.push_back(builder.getI32IntegerAttr(dim));
      }
      auto shapeType = emitc::ArrayType::get({(int64_t)shapeValues.size()},
                                             builder.getI32Type());
      auto shapeAttr = DenseElementsAttr::get(
          RankedTensorType::get({(int64_t)shapeValues.size()},
                                builder.getI32Type()),
          shapeValues);

      builder.create<emitc::GlobalOp>(builder.getUnknownLoc(),
                                      builder.getStringAttr(varName + "_shape"),
                                      TypeAttr::get(shapeType), shapeAttr,
                                      nullptr, builder.getUnitAttr(), nullptr);
      std::string rankVarName = varName + "_rank";
      int64_t rank = shape.size();
      auto rankAttr = builder.getI32IntegerAttr(rank);

      builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(rankVarName),
          mlir::TypeAttr::get(builder.getI32Type()), rankAttr, nullptr,
          builder.getUnitAttr(), nullptr);

      // value
      int64_t elementCount = tensorType.getNumElements();
      auto valueElementType = tensorType.getElementType();
      auto valueAttr = tosaConstOp.getValue();
      if (!valueAttr) {
        return;
      }

      if (auto quantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
              valueElementType)) {
        valueElementType = quantType.getStorageType();
      } else if (auto perAxisType =
                     mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                         valueElementType)) {
        valueElementType = perAxisType.getStorageType();
      }
      SmallVector<int64_t, 1> flatShape = {elementCount};

      auto flatArrayType = emitc::ArrayType::get(flatShape, valueElementType);
      auto rawTensorType = RankedTensorType::get(flatShape, valueElementType);

      auto denseAttr = llvm::cast<DenseElementsAttr>(valueAttr);
      SmallVector<Attribute> valueList;
      for (Attribute val : denseAttr.getValues<Attribute>()) {
        valueList.push_back(val);
      }
      auto flatValueAttr = DenseElementsAttr::get(rawTensorType, valueList);

      builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(varName + "_data"),
          TypeAttr::get(flatArrayType), flatValueAttr, nullptr,
          builder.getUnitAttr(), nullptr);

      globalVars.push_back(std::make_pair(varName, tensorType));
    } else if (auto constShapeOp = llvm::dyn_cast<tosa::ConstShapeOp>(op)) {
      auto valueAttr = constShapeOp.getValue();
      auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr);
      if (!denseAttr)
        return;

      auto elementType = denseAttr.getElementType();
      if (!mlir::isa<mlir::IntegerType>(elementType) &&
          !elementType.isIndex()) {
        llvm::errs() << "Unsupported element type in tosa.const_shape\n";
        return;
      }

      auto i32Type = builder.getI32Type();

      SmallVector<int32_t> dataValues;
      for (APInt val : denseAttr.getValues<APInt>()) {
        dataValues.push_back(static_cast<int32_t>(val.getSExtValue()));
      }

      llvm::SmallVector<int64_t, 1> dataShape{
          static_cast<int64_t>(dataValues.size())};
      auto dataArrayType = emitc::ArrayType::get(dataShape, i32Type);
      auto dataTensorType = RankedTensorType::get(dataShape, i32Type);
      auto dataAttr =
          DenseElementsAttr::get<int32_t>(dataTensorType, dataValues);

      int32_t shapeLength = static_cast<int32_t>(dataValues.size());
      auto shapeArrayType = emitc::ArrayType::get({1}, i32Type);
      auto shapeTensorType = RankedTensorType::get({1}, i32Type);
      auto shapeAttr =
          DenseElementsAttr::get<int32_t>(shapeTensorType, {shapeLength});

      auto shapeGlobal = builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(varName + "_shape"),
          TypeAttr::get(shapeArrayType), shapeAttr, nullptr,
          builder.getUnitAttr(), nullptr);
      llvm::SmallVector<int64_t, 4> shape;
      int64_t rank = 0;
      mlir::IntegerAttr rankAttr;

      auto resultType = constShapeOp.getResult().getType();

      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        shape = llvm::to_vector<4>(tensorType.getShape());
        rank = static_cast<int64_t>(shape.size());
        rankAttr = builder.getI32IntegerAttr(rank);
      } else {
        auto valueAttr = constShapeOp.getValue();
        if (auto denseAttr =
                mlir::dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
          rank = static_cast<int64_t>(denseAttr.getNumElements());
          shape.push_back(rank);
          rankAttr = builder.getI32IntegerAttr(rank);
        } else {
          llvm::errs() << "Error: const_shape has invalid value\n";
          return;
        }
      }

      builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(varName + "_rank"),
          mlir::TypeAttr::get(builder.getI32Type()), rankAttr, nullptr,
          builder.getUnitAttr(), nullptr);

      auto dataGlobal = builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(varName + "_data"),
          TypeAttr::get(dataArrayType), dataAttr, nullptr,
          builder.getUnitAttr(), nullptr);

      globalVars.push_back(std::make_pair(varName, dataTensorType));
    }
    // Not constOp
    else {
      if (op->getNumResults() == 0) {
        return;
      }
      auto resultType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (!resultType) {
        return;
      }

      auto shape = resultType.getShape();
      auto elemType = resultType.getElementType();

      bool isQuant = false;
      float scale = 1.0f;
      float zeroPoint = 0.0f;
      Type shapeElemType = builder.getI32Type();

      if (auto quantType =
              mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        isQuant = true;
        shapeElemType = quantType.getExpressedType();
        scale = static_cast<float>(quantType.getScale());
        zeroPoint = static_cast<float>(quantType.getZeroPoint());
      }

      SmallVector<mlir::Attribute, 4> shapeValues;
      for (int64_t d : shape) {
        shapeValues.push_back(
            builder.getI32IntegerAttr(static_cast<int32_t>(d)));
      }

      auto shapeArrayType = emitc::ArrayType::get({(int64_t)shapeValues.size()},
                                                  builder.getI32Type());

      auto shapeAttr = DenseElementsAttr::get(
          RankedTensorType::get({(int64_t)shapeValues.size()},
                                builder.getI32Type()),
          shapeValues);

      builder.create<emitc::GlobalOp>(builder.getUnknownLoc(),
                                      builder.getStringAttr(varName + "_shape"),
                                      TypeAttr::get(shapeArrayType), shapeAttr,
                                      nullptr, builder.getUnitAttr(), nullptr);
      std::string rankVarName = varName + "_rank";
      int64_t rank = shape.size();
      auto rankAttr = builder.getI32IntegerAttr(rank);

      builder.create<emitc::GlobalOp>(
          builder.getUnknownLoc(), builder.getStringAttr(rankVarName),
          mlir::TypeAttr::get(builder.getI32Type()), rankAttr, nullptr,
          builder.getUnitAttr(), nullptr);
      // rescale Multiplier & Shift
      if (auto rescaleOp = llvm::dyn_cast<tosa::RescaleOp>(op)) {
        auto multiplierAttr = rescaleOp.getMultiplierAttr();
        auto shiftAttr = rescaleOp.getShiftAttr();
        std::vector<int32_t> multiplierVec;
        for (int32_t v : multiplierAttr.asArrayRef()) {
          multiplierVec.push_back(v);
        }
        std::vector<int32_t> shiftVec;
        for (int32_t v : shiftAttr.asArrayRef()) {
          shiftVec.push_back(v);
        }
        auto arrayTypeM = emitc::ArrayType::get({(int64_t)multiplierVec.size()},
                                                builder.getI32Type());
        auto dataAttrM = DenseElementsAttr::get(
            RankedTensorType::get({(int64_t)multiplierVec.size()},
                                  builder.getI32Type()),
            llvm::ArrayRef<int32_t>(multiplierVec));

        builder.create<emitc::GlobalOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(varName + "_multiplier"),
            TypeAttr::get(arrayTypeM), dataAttrM, nullptr,
            builder.getUnitAttr(), nullptr);

        auto arrayTypeS = emitc::ArrayType::get({(int64_t)shiftVec.size()},
                                                builder.getI32Type());
        auto dataAttrS = DenseElementsAttr::get(
            RankedTensorType::get({(int64_t)shiftVec.size()},
                                  builder.getI32Type()),
            llvm::ArrayRef<int32_t>(shiftVec));

        builder.create<emitc::GlobalOp>(
            builder.getUnknownLoc(), builder.getStringAttr(varName + "_shift"),
            TypeAttr::get(arrayTypeS), dataAttrS, nullptr,
            builder.getUnitAttr(), nullptr);
      }

      if (auto reshapeOp = llvm::dyn_cast<tosa::ReshapeOp>(op)) {
        auto newShape = reshapeOp.getNewShape();

        std::vector<int32_t> shapeVec;
        shapeVec.reserve(newShape.size());
        for (int64_t dim : newShape) {
          shapeVec.push_back(static_cast<int32_t>(dim));
        }

        auto arrayType = emitc::ArrayType::get({(int64_t)shapeVec.size()},
                                               builder.getI32Type());

        auto dataAttr = DenseElementsAttr::get(
            RankedTensorType::get({(int64_t)shapeVec.size()},
                                  builder.getI32Type()),
            llvm::ArrayRef<int32_t>(shapeVec));

        builder.create<emitc::GlobalOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(varName + "_new_shape"),
            TypeAttr::get(arrayType), dataAttr, nullptr, builder.getUnitAttr(),
            nullptr);
      }

      mlir::Value result = op->getResult(0);

      size_t offsetResult = 0;
      const AllocationInfo *info = memoryPlanner.getAllocationInfo(result);
      if (info) {
        offsetResult = info->offset;
      }
      notConstGlobalVars.push_back(
          std::make_tuple(varName, offsetResult, resultType));
    }
  });

  int64_t arenaSize = static_cast<int64_t>(memoryPlanner.getPeakMemoryUsage());
  auto arenaArrayType =
      emitc::ArrayType::get({(arenaSize + 7) / 8}, builder.getI64Type());
  builder.create<emitc::GlobalOp>(
      builder.getUnknownLoc(), builder.getStringAttr("arena"),
      TypeAttr::get(arenaArrayType), nullptr, nullptr, builder.getUnitAttr());

  func::FuncOp modelInitFunc = module.lookupSymbol<func::FuncOp>("model_init");
  if (!modelInitFunc) {
    FunctionType funcType = builder.getFunctionType({}, {});
    modelInitFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                 "model_init", funcType);
    Block *entryBlock = modelInitFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
  } else {
    Block *entryBlock = &modelInitFunc.getBody().front();
    builder.setInsertionPointToEnd(entryBlock);
  }

  auto zeroIndex1 = builder.create<emitc::ConstantOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));

  auto arenaGlobal = builder.create<emitc::GetGlobalOp>(
      builder.getUnknownLoc(), arenaArrayType, builder.getStringAttr("arena"));
  auto arenaElemLValue = builder.create<emitc::SubscriptOp>(
      builder.getUnknownLoc(), emitc::LValueType::get(builder.getI64Type()),
      arenaGlobal, SmallVector<Value, 1>{zeroIndex1});
  auto alignedArenaPtr = builder.create<emitc::ApplyOp>(
      builder.getUnknownLoc(), emitc::PointerType::get(builder.getI64Type()),
      builder.getStringAttr("&"), arenaElemLValue);

  auto arenaPtr = builder.create<emitc::CastOp>(
      module.getLoc(), emitc::PointerType::get(builder.getI8Type()),
      alignedArenaPtr.getResult());

  for (const auto &[name, type] : globalVars) {
    Type element = type.getElementType();
    if (auto quant = dyn_cast<quant::QuantizedType>(element))
      element = quant.getStorageType();
    auto address =
        globalArrayAddress(builder, module.getLoc(), name + "_data", element,
                           {type.getNumElements()}, zeroIndex1);
    auto data = builder.create<emitc::CastOp>(
        module.getLoc(), emitc::PointerType::get(builder.getI8Type()), address);
    initializeTensor(builder, module.getLoc(), name, type, data, zeroIndex1);
  }
  auto initializeArenaTensor = [&](const std::string &name, size_t offset,
                                   RankedTensorType type) {
    auto offsetValue =
        builder.create<emitc::ConstantOp>(module.getLoc(), builder.getI64Type(),
                                          builder.getI64IntegerAttr(offset));
    auto data = builder.create<emitc::AddOp>(
        module.getLoc(), emitc::PointerType::get(builder.getI8Type()), arenaPtr,
        offsetValue);
    initializeTensor(builder, module.getLoc(), name, type, data, zeroIndex1);
  };
  for (const auto &[name, offset, type] : notConstGlobalVars)
    initializeArenaTensor(name, offset, type);
  for (const auto &[name, offset, type] : inputArgs)
    initializeArenaTensor(name, offset, type);

  if (modelInitFunc.getBody().front().empty() ||
      !isa<func::ReturnOp>(modelInitFunc.getBody().front().back())) {
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
  }

  // main
  func::FuncOp mainFunc = module.lookupSymbol<func::FuncOp>("main");
  auto retTy =
      emitc::PointerType::get(emitc::OpaqueType::get(context, "Tensor"));
  mainFunc.setFunctionType(FunctionType::get(
      context, mainFunc.getFunctionType().getInputs(), {retTy}));

  mainFunc.setName("model");

  Block *entryBlock = &mainFunc.getBody().front();
  builder.setInsertionPointToStart(entryBlock);

  bool invalidReturn = false;
  mainFunc.walk([&](func::ReturnOp retOp) {
    auto conversionCast =
        retOp.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    mlir::Value retVal = retOp.getOperand(0);
    // remove cast
    if (auto castOp =
            retVal.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (castOp.getOperands().size() == 1) {
        retVal = castOp.getOperands()[0];
      }
    }
    auto it = globalMap.find(retVal.getAsOpaquePointer());
    if (it == globalMap.end()) {
      retOp.emitError("Missing global mapping for return value");
      invalidReturn = true;
      return;
    }
    std::string symbolName = it->second.getSymName().str();
    auto opaqueTensorType = emitc::OpaqueType::get(context, "Tensor");
    auto lvalueType = emitc::LValueType::get(opaqueTensorType);
    auto ptrType = emitc::PointerType::get(opaqueTensorType);

    builder.setInsertionPoint(retOp);
    auto getGlobal = builder.create<emitc::GetGlobalOp>(
        retOp.getLoc(), lvalueType,
        FlatSymbolRefAttr::get(context, symbolName));
    auto addrOf = builder.create<emitc::ApplyOp>(
        retOp.getLoc(), ptrType, builder.getStringAttr("&"), getGlobal);
    builder.create<mlir::func::ReturnOp>(retOp.getLoc(), addrOf.getResult());
    retOp.erase();
    if (conversionCast && conversionCast->use_empty())
      conversionCast.erase();
  });
  if (invalidReturn)
    return failure();
  builder.setInsertionPointToEnd(module.getBody());
  builder.create<emitc::VerbatimOp>(module.getLoc(),
                                    builder.getStringAttr("}"));
  llvm::errs() << "\n";
  // Define the conversion target
  mlir::ConversionTarget target(*context);
  target.addLegalDialect<emitc::EmitCDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalDialect<tosa::TosaDialect, tensor::TensorDialect>();

  // Define the rewrite patterns
  mlir::RewritePatternSet patterns(context);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertMax_pool2DToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertAddToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertSubToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertMulToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertRescaleToEmitC);
  addPattern(patterns, *this,
             &TosaToEmitCLowering::convertTransposeConv2DToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertPadToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertConcatToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertCastToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertConstToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertConv2DToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertConstShapeToEmitC);
  addPattern(patterns, *this,
             &TosaToEmitCLowering::convertDepthwiseConv2DToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertReduceSumToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertReduceMaxToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertReshapeToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertExpToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertReciprocalToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertAvg_pool2DToEmitC);
  addPattern(patterns, *this, &TosaToEmitCLowering::convertTransposeToEmitC);

  // Apply the full conversion
  if (mlir::failed(
          mlir::applyFullConversion(module, target, std::move(patterns)))) {
    return failure();
  }

  // remove main argument
  mainFunc = module.lookupSymbol<func::FuncOp>("model");
  if (!mainFunc) {
    module.emitError("main function not found after conversion");
    return failure();
  }

  for (BlockArgument arg : mainFunc.getArguments()) {
    if (!arg.use_empty())
      return mainFunc.emitError(
          "cannot remove a model argument that still has uses");
  }
  while (mainFunc.getNumArguments())
    mainFunc.eraseArgument(mainFunc.getNumArguments() - 1);

  std::error_code error;
  llvm::ToolOutputFile opsOutput(opsFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return module.emitError("cannot open operator list: ") << error.message();
  auto &fout2 = opsOutput.os();
  for (const auto &kv : convParamMap) {
    fout2 << kv.first << " ";
    for (auto v : kv.second) {
      fout2 << v << " ";
    }
    fout2 << "\n";
  }
  fout2.flush();
  if (fout2.has_error())
    return module.emitError("cannot write operator list: ")
           << fout2.error().message();
  opsOutput.keep();

  return success();
}

namespace {
struct TosaToEmitC final
    : public PassWrapper<TosaToEmitC, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaToEmitC)
  explicit TosaToEmitC(std::string opsFile) : opsFile(std::move(opsFile)) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    TosaToEmitCLowering lowering;
    if (failed(lowering.run(getOperation(), opsFile)))
      signalPassFailure();
  }

private:
  std::string opsFile;
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToEmitC(std::string opsFile) {
  return std::make_unique<TosaToEmitC>(std::move(opsFile));
}
