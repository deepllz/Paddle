// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/hlir/framework/pir/utils.h"

#include <regex>
#include <string>
#include <unordered_map>
#include "glog/logging.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/op_mapper.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"

PD_DECLARE_string(allow_cinn_ops);
PD_DECLARE_string(deny_cinn_ops);

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

// Mapping PaddleDialect Op into CINN AST Compute register Op.
// All key names are also supported in CINN. For ops not in this
// list, we judge them by search it in CINN global Operator table.
const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd_op.full", "fill_constant"},
    {"pd_op.sum", "reduce_sum"},
    {"pd_op.max", "reduce_max"},
    {"pd_op.add", "elementwise_add"},
    {"pd_op.elementwise_pow", "pow"},
    {"pd_op.multiply", "elementwise_mul"},
    {"pd_op.maximum", "max"},
    {"pd_op.minimum", "min"},
    {"pd_op.reshape", "reshape"},
    {"pd_op.squeeze", "reshape"},
    {"pd_op.unsqueeze", "reshape"},
    {"pd_op.split_with_num", "split"},
    {"pd_op.expand", "broadcast_to"},
    {"cinn_op.generate_shape", "generate_shape"},
    {"cinn_op.broadcast", "broadcast_to"}};

namespace {
using GroupOpsVec = std::vector<::pir::Operation*>;
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";

// OpTransInfo contains informations used to detect subgraphs
// supported by the CINN compiler.
class OpTransInfo {
  using DeParamCondT =
      std::unordered_map<std::string, std::unordered_set<std::string>>;

 public:
  OpTransInfo() {}

  const DeParamCondT& deny_param_cond() const { return deny_param_cond_; }
  const std::unordered_set<std::string>& default_deny_ops() const {
    return default_deny_ops_;
  }

 private:
  DeParamCondT deny_param_cond_{{"batch_norm", {"ReserveSpace"}},
                                {"batch_norm_grad", {"ReserveSpace"}}};

  std::unordered_set<std::string> default_deny_ops_{
      "feed", "fetch", "conv2d", "conv2d_grad", "dropout", "matmul"};
};

std::unordered_set<std::string> StringSplit(const std::string& str,
                                            const std::string& delim) {
  std::regex reg(delim);
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator()};
  elems.erase("");
  return elems;
}

std::string GetDebugInfo(const std::unordered_set<std::string>& names) {
  std::string debug_info = "[";
  for (auto& name : names) {
    debug_info.append(name);
    debug_info.append(", ");
  }
  debug_info.append("]");
  return debug_info;
}

bool IsSupportForCinn(const ::pir::Operation& op);

// In case of op has some attributes generated by FullOp, it need
// implement OpPattern in pd_to_cinn_pass. Otherwise, we mark them
// as unimplement ops.
bool UnimplementOps(const ::pir::Operation& op) {
  // cinn not support uniform, the FullOp of max and min support
  // NOT generate by CINN
  if (op.isa<paddle::dialect::FullOp>()) {
    auto out = op.result(0);
    if (out.use_count() > 0) {
      return !IsSupportForCinn(*(out.first_use().owner()));
    }
  }
  return false;
}

bool HaveZeroDimInput(const ::pir::Operation& op) {
  auto HasZeroDim = [](const ::pir::Type& type) {
    auto tensor_type = type.dyn_cast<::pir::DenseTensorType>();
    return tensor_type && tensor_type.dims().size() == 0U;
  };
  // Judge for vector<Type>
  auto HasZeroDimInVT = [&](const std::vector<::pir::Type>& types) {
    for (auto& type : types) {
      if (HasZeroDim(type)) return true;
    }
    return false;
  };

  for (size_t i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    if (!value || !value.type()) continue;
    if (auto vector_type = value.type().dyn_cast<::pir::VectorType>()) {
      if (HasZeroDimInVT(vector_type.data())) return true;
    } else if (HasZeroDim(value.type())) {
      return true;
    }
  }
  return false;
}

bool AllInputDenseTensor(const ::pir::Operation& op) {
  auto IsDenseTensor = [](const ::pir::Type& type) {
    return type.isa<::pir::DenseTensorType>();
  };

  // Judge for vector<Type>
  auto IsAllDenseTensor = [&](const std::vector<::pir::Type>& types) {
    for (auto& type : types) {
      if (!IsDenseTensor(type)) return false;
    }
    return true;
  };

  for (size_t i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    if (!value || !value.type()) continue;
    if (auto vector_type = value.type().dyn_cast<::pir::VectorType>()) {
      if (!IsAllDenseTensor(vector_type.data())) return false;
    } else if (!IsDenseTensor(value.type())) {
      return false;
    }
  }

  return true;
}

bool IsRegisteredInCINN(const ::pir::Operation& op) {
  if (CompatibleInfo::OP_NAMES.find(op.name()) !=
      CompatibleInfo::OP_NAMES.end()) {
    return true;
  }
  return OpRegistry::Global()->Find(CompatibleInfo::OpName(op)) != nullptr;
}

bool IsSupportForCinn(const ::pir::Operation& op) {
  if (!AllInputDenseTensor(op) || HaveZeroDimInput(op) || UnimplementOps(op)) {
    VLOG(4) << "Found " << op.name()
            << " HaveZeroDimInput or UnimplementOps or NotAllInputDenseTensor. "
            << "So mark IsSupportForCinn: " << false;
    return false;
  }
  auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
  auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
  LOG_FIRST_N(INFO, 1) << "The allowed Cinn Ops: " << GetDebugInfo(allow_ops);
  LOG_FIRST_N(INFO, 1) << "The denied Cinn Ops: " << GetDebugInfo(deny_ops);
  // Strip the dialect, like pd_op.abs -> abs
  const auto op_name = CompatibleInfo::OpName(op);

  OpTransInfo trans_info;
  bool is_support =
      IsRegisteredInCINN(op) && !trans_info.default_deny_ops().count(op_name);
  VLOG(4) << op_name << " is_support: " << is_support
          << " IsRegisteredInCINN: " << IsRegisteredInCINN(op);
  // if the op type is registered in CINN and allow_ops is not empty, return
  // true only when it is in allow_ops
  if (!allow_ops.empty()) {
    return is_support && allow_ops.count(op_name);
  }
  // if the op type is registered in CINN and deny_ops is not empty, return
  // true only when it is not in deny_ops
  if (!deny_ops.empty()) {
    return is_support && !deny_ops.count(op_name);
  }

  // if the user doesn't set FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops,
  // return true only when it is registered in CINN
  return is_support;
}
}  // namespace

// In following cases, the op is marked SupportCinn:
// 1. its name is in OP_NAMES, like pd_op.sum;
// 2. it supports AttributeTensor but has Pattern to process it.
//    Such as cinn_op.reshape, except pd_op.reshape;
// 3. otherwise, it should be registered in OpRegistry;
bool CompatibleInfo::IsSupportCinn(const ::pir::Operation& op) {
  bool flag = IsSupportForCinn(op);
  VLOG(4) << "CompatibleInfo::IsSupportCinn of " << op.name()
          << " is: " << flag;
  return flag;
}

std::string CompatibleInfo::OpName(const ::pir::Operation& op) {
  std::string name = op.name();
  if (OP_NAMES.count(name)) {
    return OP_NAMES.at(name);
  }
  auto pos = name.find(".");
  if (pos == std::string::npos) {
    return name;
  }
  auto cinn_op_name = name.substr(pos + 1);
  VLOG(7) << "GetOpName: " << name << " -> " << cinn_op_name;
  CHECK(cinn_op_name != "")
      << "Found empty cinn_op_name, maybe you should implement OpPattern for "
      << name;
  return cinn_op_name;
}

std::string CompatibleInfo::OpFuncName(const ::pir::Operation& op) {
  std::string op_name = OpName(op);
  std::string func_name =
      cinn::common::Context::Global().NewName("fn_" + op_name);
  return func_name;
}

std::string CompatibleInfo::GroupOpsName(
    const std::vector<::pir::Operation*>& ops) {
  std::string name = "fn";
  for (auto* op : ops) {
    std::string op_name = OpName(*op);
    name += "_" + cinn::common::Context::Global().NewName(op_name);
  }
  return name;
}

std::string CompatibleInfo::ValueName(const ::pir::Value& value) {
  size_t hash_key = std::hash<::pir::Value>()(value);
  return cinn::common::Context::Global().PrettyUniqName(
      hash_key, CompatibleInfo::kNamePrefix);
}

std::vector<::pir::Value> CompatibleInfo::RealOperandSources(
    const ::pir::Operation& op) {
  if (OpMapper::Instance().has(op, MapperType::OPERAND)) {
    return OpMapper::Instance().RealOperandSources(op);
  } else {
    return op.operands_source();
  }
}

#define CASE_ATTRIBUTE(val_type, attr_type)                     \
  std::vector<val_type> res;                                    \
  for (auto element : attr_vec) {                               \
    res.push_back(element.dyn_cast<::pir::attr_type>().data()); \
  }                                                             \
  dst_attr = res;

static utils::Attribute ConvertArrayAttribute(
    const ::pir::Attribute& src_attr) {
  utils::Attribute dst_attr;
  if (src_attr.isa<paddle::dialect::IntArrayAttribute>()) {
    auto& arr = src_attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                    .data()
                    .GetData();
    std::vector<int> val(arr.begin(), arr.end());
    dst_attr = val;
  } else if (src_attr.isa<paddle::dialect::DataTypeAttribute>()) {
    auto dtype = src_attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
    dst_attr = phi::DataTypeToString(dtype);
  } else if (src_attr.isa<::pir::ArrayAttribute>()) {
    auto attr_vec = src_attr.dyn_cast<::pir::ArrayAttribute>().AsVector();
    if (attr_vec.size() > 0) {
      if (attr_vec[0].isa<::pir::Int32Attribute>()) {
        CASE_ATTRIBUTE(int32_t, Int32Attribute)
      } else if (attr_vec[0].isa<::pir::Int64Attribute>()) {
        CASE_ATTRIBUTE(int64_t, Int64Attribute)
      } else if (attr_vec[0].isa<::pir::BoolAttribute>()) {
        CASE_ATTRIBUTE(bool, BoolAttribute)
      } else if (attr_vec[0].isa<::pir::FloatAttribute>()) {
        CASE_ATTRIBUTE(float, FloatAttribute)
      } else if (attr_vec[0].isa<::pir::DoubleAttribute>()) {
        CASE_ATTRIBUTE(double, DoubleAttribute)
      } else {
        LOG(FATAL) << "only support bool/int32/int64/float/double attribute in "
                      "ArrayAttribute";
      }
    }
  } else {
    LOG(FATAL) << "unknown Attribute: " << src_attr;
  }
  return dst_attr;
}
#undef CASE_ATTRIBUTE

#define CASE_SINGLE_ATTR(attr_type, func)               \
  else if (src_attr.isa<::pir::attr_type>()) dst_attr = \
      src_attr.dyn_cast<::pir::attr_type>().func();

utils::Attribute CompatibleInfo::ConvertAttribute(
    const ::pir::Attribute& src_attr) {
  utils::Attribute dst_attr;
  if (src_attr.isa<::pir::BoolAttribute>())
    dst_attr = src_attr.dyn_cast<::pir::BoolAttribute>().data();
  CASE_SINGLE_ATTR(FloatAttribute, data)
  CASE_SINGLE_ATTR(DoubleAttribute, data)
  CASE_SINGLE_ATTR(Int32Attribute, data)
  CASE_SINGLE_ATTR(Int64Attribute, data)
  CASE_SINGLE_ATTR(StrAttribute, AsString)
  else if (src_attr.isa<::pir::shape::SymbolAttribute>()) return dst_attr;
  else dst_attr = ConvertArrayAttribute(src_attr);  // NOLINT
  return dst_attr;
}
#undef CASE_SINGLE_ATTR

utils::AttributeMap CompatibleInfo::ConvertAttributes(
    const ::pir::Operation& op) {
  auto& src_attrs = op.attributes();
  utils::AttributeMap dst_attrs;
  for (auto& item : src_attrs) {
    VLOG(4) << "deal with " << item.first;
    if (item.first == ::pir::kStopGradientAttrName) {
      continue;
    } else if (item.second.isa<paddle::dialect::PlaceAttribute>()) {
      auto is_cpu =
          item.second.dyn_cast<paddle::dialect::PlaceAttribute>().data() ==
          phi::CPUPlace();
      dst_attrs["force_cpu"] = is_cpu;
    } else {
      dst_attrs[item.first] = std::move(ConvertAttribute(item.second));
    }
  }

  if (OpMapper::Instance().has(op, MapperType::ATTRIBUTE)) {
    OpMapper::Instance().AppendVariantAttrs(op, dst_attrs);
  }
  VLOG(4) << "dst_attrs.size(): " << dst_attrs.size();
  return dst_attrs;
}

#define CASE_TYPE(src, dst) \
  else if (type.isa<::pir::src>()) return cinn::common::dst();

cinn::common::Type CompatibleInfo::ConvertIRType(::pir::Type type) {
  if (type.isa<::pir::BFloat16Type>()) return cinn::common::BF16();
  CASE_TYPE(Float16Type, F16)
  CASE_TYPE(Float32Type, F32)
  CASE_TYPE(Float64Type, F64)
  CASE_TYPE(Int8Type, I8)
  CASE_TYPE(UInt8Type, UI8)
  CASE_TYPE(Int16Type, I16)
  CASE_TYPE(Int32Type, I32)
  CASE_TYPE(Int64Type, I64)
  CASE_TYPE(IndexType, I32)
  CASE_TYPE(BoolType, UI1)

  LOG(FATAL) << "unknown ir::Type " << type;
}
#undef CASE_TYPE

int CompatibleInfo::ShapeProduct(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

OpPatternKind CompatibleInfo::OpKind(const ::pir::Operation& op) {
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  auto op_name = CompatibleInfo::OpName(op);
  if (op_name == "generate_shape") {
    return hlir::framework::kNonFusible;
  }
  const hlir::framework::Operator* cinn_op = Operator::Get(op_name);
  CHECK(op_pattern_dict.Find(cinn_op));
  auto kind = op_pattern_dict[cinn_op];
  if (kind == hlir::framework::kBroadcast) {
    // As binary op was defined as broadcast, actually it should be
    // element-wise. See fusion_hepler_base.h for detail.
    if (op_name != "broadcast_to") {
      kind = hlir::framework::kElementWise;
    }
  }
  VLOG(4) << op_name << " OpPatternKind: " << kind;
  return kind;
}

std::vector<int> CompatibleInfo::ValueShape(const ::pir::Value& value) {
  auto& dim = value.type().dyn_cast<::pir::DenseTensorType>().dims();
  return ::common::vectorize<int>(dim);
}

std::vector<int64_t> GetBroadcastAxis(const phi::DDim& in_shape,
                                      const std::vector<int64_t>& out_shape) {
  std::vector<int64_t> broadcast_axes(in_shape.size(), 0);
  auto in_shape_size = in_shape.size();
  if (in_shape_size >= 1) {
    for (int i = 1; i <= in_shape_size; ++i) {
      broadcast_axes[in_shape_size - i] = out_shape.size() - i;
    }
  }

  return broadcast_axes;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
