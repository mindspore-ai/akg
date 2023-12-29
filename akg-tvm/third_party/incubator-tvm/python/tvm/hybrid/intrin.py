# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# 2019.12.30 - Add file intrin.py. Supporting arbitrary scalar intrinsics in hybrid script.

""" Supporting arbitrary scalar intrinsics in hybrid script. """

from tvm import make as _make
from tvm import expr as _expr

from tvm.hybrid import calls

from tvm.hybrid.runtime import HYBRID_GLOBALS
from tvm.hybrid.util import _AssertCallable, _internal_assert_arg_type, _internal_assert

# To distinguish between the intrin registered via _register_intrin
# and the other methods in calls.py.
_REGISTER_INTRIN = set([])


def _register_intrin(token, intrin):
    """Add an intrin into calls dynamically."""
    _internal_assert(not hasattr(calls, token),
                     '{} already defined!'.format(token))
    _REGISTER_INTRIN.add(token)
    if isinstance(intrin, _TensorIntrin):
        setattr(calls, token, intrin)
    else:
        setattr(calls, token, lambda func_id, args: intrin(*args))


def _unregister_intrin(token):
    """Remove the intrin added by _register_intrin."""
    _internal_assert(token in _REGISTER_INTRIN, '{} not defined!'.format(token))
    _REGISTER_INTRIN.remove(token)
    delattr(calls, token)


# To distinguish between intrins and other globals in HYBRID_GLOBALS.
HYBRID_GLOBALS_INTRINS = set([])


def _register_intrin_emulate(token, emulate):
    """Register the emulation of an intrinsic into the hybrid runtime."""
    _internal_assert(token not in HYBRID_GLOBALS,
                     '{} already defined!'.format(token))
    HYBRID_GLOBALS_INTRINS.add(token)
    HYBRID_GLOBALS[token] = emulate


def _unregister_intrin_emulate(token):
    """Unregister the emulation of an intrinsic into the hybrid runtime."""
    _internal_assert(token in HYBRID_GLOBALS_INTRINS,
                     '{} not defined!'.format(token))
    HYBRID_GLOBALS_INTRINS.remove(token)
    del HYBRID_GLOBALS[token]


class IntrinDef(object):
    """ A struct for intrinsic definition. One needs to specify the token
    of this intrinsic used in the hybrid script, the intrin call and the
    function to simulate such intrinsic.
    """

    def __init__(self, token, intrin, emulate):
        """Init. with intrinsic definition.

        Parameters
        ----------
        token: A string.
            The intrinsic name/token that is used in the hybrid script.

        intrin:
            The actual intrinsic call (e.g.,
            lambda x: tvm.call_pure_intrin(x.dtype, 'my_intrin', x)).

        emulate:
            The function to emulate intrin during the hybrid emulation runtime.
        """
        _internal_assert_arg_type(token, 'token', [str])
        _internal_assert_arg_type(intrin, 'intrin', [_AssertCallable])
        _internal_assert_arg_type(emulate, 'emulate', [_AssertCallable])
        self._token = token
        self._intrin = intrin
        self._emulate = emulate


def _iter_intrins(func, attr1, attr2=None, intrinsics=None):
    """A wrapper for _(un)?patch_intrins_[to|from]_* to reuse
    the branching and loop.
    """
    if intrinsics is not None:
        if isinstance(intrinsics, list):
            for intrin_def in intrinsics:
                if attr2 is None:
                    func(getattr(intrin_def, attr1))
                else:
                    func(getattr(intrin_def, attr1), getattr(intrin_def, attr2))
        else:
            if attr2 is None:
                func(getattr(intrinsics, attr1))
            else:
                func(getattr(intrinsics, attr1), getattr(intrinsics, attr2))


def _patch_intrins_to_runtime(intrinsics=None):
    """Register intrinsics to runtime."""
    _iter_intrins(_register_intrin_emulate,
                  '_token',
                  '_emulate',
                  intrinsics=intrinsics)


def _unpatch_intrins_from_runtime(intrinsics=None):
    """Remove intrinsics from runtime."""
    _iter_intrins(_unregister_intrin_emulate, '_token', intrinsics=intrinsics)


def _patch_intrins_to_calls(intrinsics=None):
    """Register intrinsics to calls."""
    _iter_intrins(_register_intrin, '_token', '_intrin', intrinsics=intrinsics)


def _unpatch_intrins_from_calls(intrinsics=None):
    """Remove intrinsics from calls."""
    _iter_intrins(_unregister_intrin, '_token', intrinsics=intrinsics)


class _TensorIntrin(object):

    def __init__(self, body, input_buffers, output_buffers):
        self._body = body
        self._input_buffers = input_buffers
        self._output_buffers = output_buffers

    def __call__(self,
                 *args,
                 input_buffer_map=None,
                 output_buffer_map=None,
                 input_region_map=None,
                 output_region_map=None):
        # Update the mappings from the parser.
        num_inputs = len(self._input_buffers)
        num_outputs = len(self._output_buffers)
        in_tensors = args[:num_inputs]
        out_tensors = args[num_inputs:num_inputs + num_outputs]
        in_regions = args[num_inputs + num_outputs:2 * num_inputs + num_outputs]
        out_regions = args[2 * num_inputs + num_outputs:]

        def update_mapping(keys, values, mapping):
            for k, v in zip(keys, values):
                mapping[k] = v

        update_mapping(in_tensors, self._input_buffers, input_buffer_map)
        update_mapping(out_tensors, self._output_buffers, output_buffer_map)
        update_mapping(in_tensors, in_regions, input_region_map)
        update_mapping(out_tensors, out_regions, output_region_map)
        # Return the intrin Stmt.
        body = self._body
        if isinstance(body, _expr.Expr):
            body = _make.Evaluate(body)
        return body


def decl_tensor_intrin(fcompute, input_buffers, output_buffers):
    body = fcompute(input_buffers, output_buffers)
    return _TensorIntrin(body, input_buffers, output_buffers)
