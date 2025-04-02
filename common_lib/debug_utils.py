# Copyright 2023 The Languini Authors.
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

import torch
import numpy as np


def check(tensor, shape):
    """ Checks the shape of the tensor for better code redability and bug prevention. """
    if tensor is None:
        return

    assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: tensor is not torch.Tensor!"
    tensor = tensor.detach()  # necessary for torch.compile
    tensor_shape = list(tensor.shape)

    assert isinstance(shape, list) or isinstance(shape, tuple), "shape arg has to be a tuple/list!"
    assert len(shape) == len(tensor_shape), f"SHAPE GUARD: tensor shape {tensor_shape} not the same length as {shape}"

    for idx, (a, b) in enumerate(zip(tensor_shape, shape)):
        if b <= 0:
            continue  # ignore -1 sizes
        else:
            assert a == b, f"SHAPE GUARD: at pos {str(idx)}, tensor shape {tensor_shape} does not match {shape}"