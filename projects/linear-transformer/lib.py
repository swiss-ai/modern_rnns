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

import math
import torch
import torch.nn.functional as F

from torch import nn
from common_lib.debug_utils import check

from collections import OrderedDict


class Event(object):
    """The Event is the base class for all events that are dispatched from any
    transformer module.

    This class defines only the basic attributes of an event without any
    payload.

    Arguments
    ---------
        source: torch.nn.Module instance that dispatched this event
    """
    def __init__(self, source):
        self.source = source

class EventFilter(object):
    """EventFilter instances are predicates (ie functions that return True or
    False) to be used with an event dispatcher for filtering event
    instances.

    The main benefit from using raw functions is that an EventFilter composes
    very easily using operators such as &, |, ~.

    Example
    --------

        event_filter = AttentionEvent | layer_name_contains("layers.1")
        event_filter = from_layer(transformer.layers[2].attention)
        event_filter = (
            AttentionEvent &
            lambda ev: torch.isnan(ev.attention_matrix).any()
        )
    """
    def __call__(self, event):
        raise NotImplementedError()

    def _to_event_filter(self, other):
        if isinstance(other, EventFilter):
            return other
        if isinstance(other, type) and issubclass(other, Event):
            return event_class(other)
        if callable(other):
            return CallableEventFilter(other)

        return NotImplemented

    def __and__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) and other(ev))

    def __rand__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) and self(ev))

    def __or__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: self(ev) or other(ev))

    def __ror__(self, other):
        other = self._to_event_filter(other)
        if other is NotImplemented:
            return other
        return CallableEventFilter(lambda ev: other(ev) or self(ev))

    def __invert__(self):
        return CallableEventFilter(lambda ev: not self(ev))

class CallableEventFilter(EventFilter):
    """Wrap a function with an EventFilter object."""
    def __init__(self, event_filter):
        self._event_filter = event_filter

    def __call__(self, event):
        return self._event_filter(event)
def event_class(klass):
    """Select events that are instances of `klass`.

    Arguments
    ---------
        klass: A class to check the event instance against

    Returns
    -------
        An instance of EventFilter
    """
    return CallableEventFilter(lambda ev: isinstance(ev, klass))
class EventDispatcher(object):
    """An EventDispatcher is a simple way to implement an observer pattern for
    loose coupling of components. In our case it is used so that the internals
    of large neural networks can communicate with the outside world in an
    agnostic and efficient way.

    Example usage
    -------------

        from fast_transformers.events import EventDispatcher, AttentionEvent
        from fast_transformers.events.filters import \
            layer_name_contains

        def attention_event_handler(event):
            print(event.attention_matrix)

        ed = EventDispatcher()
        ed.listen(AttentionEvent, attention_event_handler)
        ed.listen(
            AttentionEvent & layer_name_contains("layers.12"),
            attention_event_handler
        )
    """
    _dispatchers = {}

    def __init__(self):
        self._listeners = OrderedDict()

    def listen(self, event_filter, event_handler):
        """Add an event handler for the events that pass the event filter.

        Arguments
        ---------
            event_filter: callable or Event class to define for which events
                          this handler will be called
            event_handler: callable that accepts an instance of Event
        """
        if isinstance(event_filter, type) and issubclass(event_filter, Event):
            event_filter = event_class(event_filter)

        self._listeners[event_handler] = event_filter

    def remove(self, event_handler):
        """Remove the event_handler from the listeners so that no more events
        are dispatched to this handler."""
        self._listeners.pop(event_handler, None)

    def clear(self):
        """Remove all listeners from the event dispatcher."""
        self._listeners.clear()

    def dispatch(self, event):
        """Dispatch an event to the listeners.

        Arguments
        ---------
            event: Event instance
        """
        for event_handler, event_filter in self._listeners.items():
            if event_filter(event):
                event_handler(event)

    @classmethod
    def get(cls, key=""):
        """Factory method for creating global event dispatchers for loosely
        coupling parts of a larger codebase.

        Since global objects are a complete antipattern, we suggest that this
        is only used to set a default value for an event dispatcher passed as
        an argument.

        Argument
        --------
            key: A key to uniquely identify a dispatcher or an instance of a
                 dispatcher to be returned as is
        """
        if isinstance(key, cls):
            return key
        if key not in cls._dispatchers:
            cls._dispatchers[key] = cls()
        return cls._dispatchers[key]

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)

elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU)
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    def __init__(self, h_dim, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.weight = nn.Parameter(torch.ones(h_dim))
        self.bias = nn.Parameter(torch.zeros(h_dim))

    def forward(self, x, log=None):
        bsz, seqlen, _ = x.shape
        check(x, (bsz, seqlen, self.h_dim))

        y = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        return y

class MLP(torch.nn.Module):
    def __init__(self, h_dim, mlp_dim, n_layers, name):
        super().__init__()
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers
        self.name = name
        self.activation_fn = gelu

        self.c_fc = nn.Linear(h_dim, mlp_dim, bias=True)
        torch.nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.c_fc.bias)

        self.c_proj = nn.Linear(mlp_dim, h_dim, bias=True)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        torch.nn.init.zeros_(self.c_proj.bias)

    def forward(self, x, log=None):
        bsz, seq_len, _ = x.shape

        check(x, (bsz, seq_len, self.h_dim))

        x = self.c_fc(x)
        check(x, (bsz, seq_len, self.mlp_dim))

        x = self.activation_fn(x)

        x = self.c_proj(x)
        check(x, (bsz, seq_len, self.h_dim))

        return x

    def __repr__(self):
        return f"MLP(h_dim={self.h_dim}, mlp_dim={self.mlp_dim}, activation_fn={self.activation_fn}, name={self.name})"


class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()



class Block(nn.Module):
    def __init__(self, seq_len, h_dim, mlp_dim, head_dim, n_heads, n_layers, block_length, name):
        super().__init__()
        self.name = name
        self.h_dim = h_dim
        self.seq_len = seq_len

        self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")

        self.rnn = LinearAttention(seq_len=seq_len, h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                              block_length=block_length,
                                              name=f"{self.name}.LinearAttention")

        self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
        self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

    def forward(self, x, state, log=None):
        bsz, _, _ = x.shape
        check(x, (bsz, self.seq_len, self.h_dim))

        ln_x = self.ln1(x, log=log)
        rnn_x, new_state = self.rnn(f_in=ln_x, i_in=ln_x, z_in=ln_x, o_in=ln_x, state=state, log=log)
        x = x + rnn_x

        mlp_x = self.mlp(self.ln2(x, log=log), log=log)
        x = x + mlp_x

        return x, new_state

class TransformerEncoderLayer(nnModule):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)
