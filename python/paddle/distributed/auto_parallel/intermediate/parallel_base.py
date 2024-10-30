#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.nn import Layer
from paddle.optimizer import Optimizer


class ParallelModelBase(Layer):
    def __init__(self, model):
        super().__init__()
        self.pp_parallelizer = None
        self.tp_parallelizer = None
        self.sharding_parallelizer = None

        if isinstance(model, ParallelModelBase):
            self.pp_parallelizer = model.pp_parallelizer
            self.tp_parallelizer = model.tp_parallelizer
            self.sharding_parallelizer = model.sharding_parallelizer

        if isinstance(model, ParallelModelBase):
            self.model = model.model
        else:
            self.model = model

        self.is_parallelized = False

    def parallelize_model(self):
        if self.pp_parallelizer is not None:
            assert callable(self.pp_parallelizer)
            self.model = self.pp_parallelizer(self.model)

        if self.tp_parallelizer is not None:
            assert callable(self.tp_parallelizer)
            self.model = self.tp_parallelizer(self.model)

        if self.sharding_parallelizer is not None:
            assert callable(self.sharding_parallelizer)
            self.model = self.sharding_parallelizer(self.model)

    def forward(self, *args):
        if not self.is_parallelized:
            self.parallelize_model()
            self.is_parallelized = True
        return self.model(*args)


class ParallelOptimizer:
    def __init__(self, optimizer=None, model=None) -> None:
        super().__init__()
        assert isinstance(
            optimizer, (ParallelOptimizer, Optimizer)
        ), "optimizer must be type of Optimizer or ParallelOptimizer"

        assert isinstance(model, ParallelModelBase)

        if isinstance(optimizer, ParallelOptimizer):
            self.optimizer = optimizer.optimizer
        else:
            self.optimizer = optimizer

        self.parallized_model = model
        self.is_parallelized = False

    def parallelize_optimizer(self):
        # 1.replace optimizer parameters with self.parallized_model.paraleters()
        self.optimizer._parameter_list = (
            self.parallized_model.model.parameters()
        )

        # 2.wrap with shard_optimizer
        level = getattr(self.parallized_model, 'level', None)
        mesh = fleet.auto.get_mesh()
        if level == "os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage1(mesh)
            )
        elif level == "os_g":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage2(mesh)
            )
        elif level == "p_g_os":
            self.optimizer = dist.shard_optimizer(
                self.optimizer, dist.ShardingStage3(mesh)
            )
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer)

    def step(self):
        if not self.is_parallelized:
            self.parallelize_optimizer()
            self.is_parallelized = True
        self.optimizer.step()

    def clear_grad(self):
        self.optimizer.clear_grad()
