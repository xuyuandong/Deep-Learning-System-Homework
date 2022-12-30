"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        print('up:', value.shape)
        #print('up:', value.grad)
        return [value]
    elif isinstance(value, Module):
        print('upM:', value)
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            #print('upDict:', k, v)
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            #print('upTL:', v)
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        wshape = (self.in_features, self.out_features)
        bshape = (1, self.out_features)
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, shape=wshape, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(self.out_features, 1, shape=bshape, device=device, dtype=dtype, requires_grad=bias))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        if self.bias.requires_grad:
            return y + ops.broadcast_to(self.bias, y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        from functools import reduce
        batch = X.shape[0]
        dims = reduce(lambda x,y : x*y, X.shape[1:])
        return ops.reshape(X, (batch, dims))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1.0)**(-1)
        # return 0.5 + 0.5 * ops.tanh(0.5 * x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        print('seq---->', x.shape)
        #print(x)
        for m in self.modules:
            print(m)
            x = m(x)
        print('<----seq', x.shape)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[1]        
        losses = ops.logsumexp(logits, axes = 1) - ops.summation(logits * init.one_hot(n, y, device=y.device), axes = 1)
        return ops.summation(losses / losses.shape[0]) 
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(1, dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(1, dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size = x.shape[0]
            mean = ops.summation(x, axes = 0) / batch_size
            mean = mean.reshape((1, mean.shape[0]))

            var = ops.summation((x - ops.broadcast_to(mean, x.shape))**2, axes = 0) / batch_size
            var = var.reshape((1, var.shape[0]))

            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data

            y = (x - ops.broadcast_to(mean, x.shape)) / ops.broadcast_to((var + self.eps)**0.5, x.shape) 
            return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        else:
            y = (x - ops.broadcast_to(self.running_mean, x.shape)) / ops.power_scalar(ops.broadcast_to(self.running_var + self.eps, x.shape), 0.5)
            return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        ### END YOUR SOLUTION



class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[1]
        mean = ops.summation(x, axes = 1) / self.dim
        var = ops.summation((x - ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape))**2, axes=1) / self.dim
        var = ops.broadcast_to(var.reshape((x.shape[0], 1)), x.shape)
        y = (x - ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape)) / (var + self.eps)**0.5
        return y * ops.broadcast_to(self.weight, y.shape) + ops.broadcast_to(self.bias, y.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        return x * init.randb(*x.shape, p = 1 - self.p) / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print(self.fn)
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        in_features = self.in_channels * self.kernel_size**2
        out_features = self.out_channels * self.kernel_size**2
        kernel_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, shape=kernel_shape, device=device, dtype=dtype, requires_grad=True))
        
        interval = 1.0/(self.in_channels * self.kernel_size**2)**0.5
        self.bias = Parameter(init.rand(self.out_channels, low=-interval, high=interval, device=device, dtype=dtype, requires_grad=bias))
        self.bias.requires_grad = bias
        
        self.padding = (self.kernel_size - 1)//2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # input X ~ NCHW
        print('CNN:', self.in_channels, self.out_channels, self.kernel_size, self.stride, 'pad=',self.padding)
        print('CXshape:', x.shape)
        x = x.transpose((1,2)).transpose((2,3))
        y = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        print('CYshape:', y.shape)
        print('CBias:', self.bias.shape)
        # now y ~ NHWC
        if self.bias.requires_grad:
            y = y + ops.broadcast_to(self.bias.reshape((1,1,1,self.out_channels)), y.shape)
        return y.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = np.sqrt(1./hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=bias))
        self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=bias))
        self.active_func = Tanh() if nonlinearity == 'tanh' else ReLU()
        self.hidden_size = hidden_size
        self.bias = bias
        print('cell init bias', bias, self.bias_ih.requires_grad)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        #print('RNNCell', X.shape, self.hidden_size, self.bias_ih.requires_grad, h if h is None else h.shape)
        y1 = X @ self.W_ih
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        y2 = h @ self.W_hh

        if self.bias:
            b1 = ops.broadcast_to(self.bias_ih.reshape((1, self.hidden_size)), y1.shape)
            b2 = ops.broadcast_to(self.bias_hh.reshape((1, self.hidden_size)), y1.shape)
            return self.active_func(y1 + y2 + b1 + b2)
        
        return self.active_func(y1 + y2)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        print('rnn layers', len(self.rnn_cells))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        
        ### BEGIN YOUR SOLUTION
        xseq = ops.split(X, 0)
        h0s = ops.split(h0, 0) if h0 is not None else [None]*len(self.rnn_cells)

        state = []
        for i, cell in enumerate(self.rnn_cells):
            oseq = []
            h = h0s[i]
            for x in xseq:
                h = cell(x, h)
                oseq.append(h)
            xseq = oseq
            state.append(xseq[-1].detach())

        return ops.stack(xseq, 0), ops.stack(state, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=bias))
        self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=bias))
        self.bias = bias
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        #print('LSTMCell', X.shape, self.hidden_size, self.bias)
        
        if h is not None:
            h0, c0 = h
        else:
            h0 = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = 0

        xW = X @ self.W_ih
        hW = h0 @ self.W_hh
        if self.bias:
            xW = xW + ops.broadcast_to(self.bias_ih.reshape((1, 4*self.hidden_size)), xW.shape)
            hW = hW + ops.broadcast_to(self.bias_hh.reshape((1, 4*self.hidden_size)), xW.shape)

        bs = X.shape[0]
        y = (xW + hW).reshape((bs, 4, self.hidden_size))
        #print('LSTMCell y', y.shape)

        sigmoid = Sigmoid()
        tanh = Tanh()
        into, forget, gate, out = ops.split(y, 1)
        into = sigmoid(into)
        forget = sigmoid(forget)
        gate = tanh(gate)
        out = sigmoid(out)
        
        cnew = forget * c0 + into * gate
        hnew = out * tanh(cnew)
        return (hnew, cnew)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        print('lstm layers', len(self.lstm_cells))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        xseq = ops.split(X, 0)
        num_layers = len(self.lstm_cells)
        hseq = [None] * num_layers
        if h is not None:
            h0, c0 = h
            h0s = ops.split(h0, 0)
            c0s = ops.split(c0, 0)
            hseq = [(h0s[i], c0s[i]) for i in range(num_layers)]

        hstate = []
        cstate = []
        for i, cell in enumerate(self.lstm_cells):
            oseq = []
            hc = hseq[i]
            for x in xseq:
                hc = cell(x, hc)
                oseq.append(hc[0])
            xseq = oseq
            hstate.append(hc[0].detach())
            cstate.append(hc[1].detach())

        output = ops.stack(xseq, 0)
        hn = ops.stack(hstate, 0)
        cn = ops.stack(cstate, 0)
        return (output, (hn, cn))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x = init.one_hot(self.num_embeddings, x, device=self.weight.device)
        #print('emb:', x.shape)
        seq_len, bs, num_embs = x.shape
        y =  x.reshape((seq_len*bs, num_embs)) @ self.weight
        return y.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION
