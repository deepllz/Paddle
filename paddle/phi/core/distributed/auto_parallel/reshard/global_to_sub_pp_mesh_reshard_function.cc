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

#include "paddle/phi/core/distributed/auto_parallel/reshard/global_to_sub_pp_mesh_reshard_function.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi {
namespace distributed {

std::vector<ProcessMesh> GetSubPPMesh(const ProcessMesh& process_mesh) {
  const std::vector<int64_t>& shape = process_mesh.shape();
  const std::vector<int64_t>& process_ids = process_mesh.process_ids();
  const std::vector<std::string>& dim_names = process_mesh.dim_names();
  int64_t total_process_num = process_ids.size();
  int64_t sub_process_num = total_process_num / shape[0];
  std::vector<int64_t> sub_process_mesh_shape(shape.begin() + 1, shape.end());
  std::vector<std::string> sub_process_mesh_dim_names(dim_names.begin() + 1,
                                                      dim_names.end());

  std::vector<ProcessMesh> sub_process_meshes;
  for (int i = 0; i < shape[0]; ++i) {
    int64_t start_position = i * sub_process_num;
    int64_t end_position = start_position + sub_process_num;
    std::vector<int64_t> sub_process_ids(process_ids.begin() + start_position,
                                         process_ids.begin() + end_position);

    sub_process_meshes.emplace_back(ProcessMesh(
        sub_process_mesh_shape, sub_process_ids, sub_process_mesh_dim_names));
  }
  return sub_process_meshes;
}

bool GlobalToSubPPMeshReshardFunction::IsSuitable(
    const DistTensor& in, const TensorDistAttr& out_dist_attr) {
  const TensorDistAttr& in_dist_attr = in.dist_attr();
  // 1. first dimension(pp) must be replicated
  RESHARD_SHORTCUT_IF_FALSE(in_dist_attr.is_replicated(0));
  // 2. out mesh is the value of a certain dimension of global mesh
  // e.g. global_mesh = [[1, 2], [3, 4]], out_mesh = [1, 2] or [3, 4]
  //      global_mesh = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  //      out_mesh = [[1, 2], [3, 4]] or [[5, 6], [7, 8]]

  const ProcessMesh& in_process_mesh = in_dist_attr.process_mesh();
  const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();

  RESHARD_SHORTCUT_IF_FALSE(in_process_mesh.ndim() ==
                            out_process_mesh.ndim() + 1);

  std::vector<ProcessMesh> sub_process_meshes = GetSubPPMesh(in_process_mesh);
  for (const ProcessMesh& sub_mesh : sub_process_meshes) {
    if (out_process_mesh == sub_mesh) {
      return true;
    }
  }
  return false;
}

void GlobalToSubPPMeshReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                                            const DistTensor& in,
                                            const TensorDistAttr& out_dist_attr,
                                            DistTensor* out) {
  VLOG(3) << "Call GlobalToSubPPMeshReshardFunction Eval";
  const DenseTensor& in_dense_value = in.value();
  // const TensorDistAttr& in_dist_attr = in.dist_attr();
  // const ProcessMesh& in_process_mesh = in_dist_attr.process_mesh();
  // const std::vector<int64_t>& in_process_ids = in_process_mesh.process_ids();
  // const ProcessMesh& out_process_mesh = out_dist_attr.process_mesh();
  // const std::vector<int64_t>& out_process_ids =
  // out_process_mesh.process_ids(); DataType dtype = in.dtype();

  // if (IsCurRankInMesh(out_process_mesh)) {
  SetValue(out, in_dense_value);
  SetDistProps(out, in.dims(), out_dist_attr);
  // } else {

  // }
}

}  // namespace distributed
}  // namespace phi
