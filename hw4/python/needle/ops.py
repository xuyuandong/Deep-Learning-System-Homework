"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, - out_grad * lhs / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axies = [i for i in range(len(a.shape))]
        x, y = self.axes if self.axes else (axies[-1], axies[-2])
        axies[x], axies[y] = axies[y], axies[x]
        return a.permute(axies).compact()
        #return array_api.swapaxes(a, x, y)     
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)   
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        exdims = len(self.shape) - len(a.shape)
        axis = [i for i in range(exdims)]
        for i, x in enumerate(a.shape):
            if x < self.shape[i + exdims]:
                axis.append(i + exdims)
        #print('bctg', self.shape, a.shape, out_grad.shape, axis)
        if len(axis) == 0:
            return reshape(out_grad, a.shape)
        return reshape(summation(out_grad, tuple(axis)), a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


def expand_dim(axes, from_shape, to_shape):
    ex_shape = [1] * len(to_shape)
    axis = []
    if axes is not None:
        axis = [axes] if not isinstance(axes, tuple) else list(axes)
    n = 0
    for i, _ in enumerate(ex_shape):
        if axes is not None and (i not in axis):
            ex_shape[i] = from_shape[n]
            n += 1
    return ex_shape

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.device is not array_api.default_device():
            if isinstance(self.axes, (tuple, list)) and len(self.axes) > 1:
                for x in self.axes:
                    a = a.sum(axis=x, keepdims=True)  ##TODO: to be fixed, this is wrong and tricky and temporal
                return a
        return a.sum(axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #ex_out_grad = array_api.expand_dims(out_grad, self.axes)
        a = node.inputs[0]
        exshape = expand_dim(self.axes, from_shape=out_grad.shape, to_shape=a.shape)
        #print('sumg', self.axes, a.shape, out_grad.shape, exshape)
        return broadcast_to(reshape(out_grad, exshape), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, transpose(rhs))
        rgrad = matmul(transpose(lhs), out_grad)
        lsize = len(out_grad.shape) - len(lhs.shape)
        rsize = len(out_grad.shape) - len(rhs.shape)
        if lsize > 0:
            lgrad = summation(lgrad, axes=tuple([i for i in range(lsize)]))
        if rsize > 0:
            rgrad = summation(rgrad, axes=tuple([i for i in range(rsize)]))
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = out_grad * (node.inputs[0].realize_cached_data()>0) 
        return y
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(self.axes)
        exshape = expand_dim(self.axes, maxz.shape, Z.shape)
        x = Z - array_api.broadcast_to(array_api.reshape(maxz, exshape), Z.shape)
        #return array_api.log(array_api.sum(array_api.exp(x), self.axes)) + maxz
        return array_api.log(array_api.exp(x).sum(self.axes)) + maxz
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        maxz = Z.realize_cached_data().max(self.axes)
        exshape = expand_dim(self.axes, maxz.shape, Z.shape)
        x = Z - array_api.broadcast_to(array_api.reshape(maxz, exshape), Z.shape)
        return broadcast_to(reshape(out_grad / summation(exp(x), self.axes), exshape), Z.shape) * exp(x)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        y = tanh(a)
        return out_grad - out_grad * y *y
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = len(args)
        shape = list(args[0].shape)
        shape.insert(0, n)
        #print('stackshape:', shape)
        result = NDArray.make(shape, device=args[0].device).reshape((n, args[0].size))
        #print('astrides', args[0].strides)

        for i in range(n):
            result[i:i+1, :] = args[i].reshape((1, args[i].size))

        #print('ar', result)
        result = result.reshape(shape)
        #print('arreshape', result)
        #print('arstrdes', result.strides)

        if self.axis != 0:
            axies = [i for i in range(len(shape))]
            axies[:self.axis] = axies[1:self.axis+1]
            axies[self.axis] = 0
            #print('axies', axies)
            result = result.permute(axies)
            #print('fstrides',result.strides)
        #print('final', result)
        return result.compact()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        n = A.shape[self.axis]
        if self.axis != 0:
            axies = [i for i in range(len(shape))]
            axies[1:self.axis+1] = axies[0:self.axis]
            axies[0] = self.axis
            src = A.permute(axies)
        else:
            src = A

        results = []
        subshape = list(src.shape)[1:]
        src = src.compact().reshape((n, A.size//n))
        for i in range(n):
            #arr = NDArray.make((1,A.size//n), device=A.device)
            arr = src[i:i+1, :]
            results.append(arr.compact().reshape(subshape))

        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = [s for s in a.shape]
        idxs = [slice(None) for i in range(a.ndim)]
        for i in self.axes:
            if i < a.ndim:
                newshape[i] = a.shape[i] * (self.dilation + 1)
                idxs[i] = slice(None, None, self.dilation + 1)
        result = NDArray.make(tuple(newshape), device=a.device)
        result.fill(0)
        result[tuple(idxs)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        #newshape = [s for s in a.shape]
        idxs = [slice(None) for i in range(a.ndim)]
        for i in self.axes:
            #newshape[i] = a.shape[i] // (self.dilation + 1)
            idxs[i] = slice(None, None, self.dilation + 1)
        return a[tuple(idxs)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        axes = ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0))
        A = A.pad(axes)

        N,H,W,Cin = A.shape
        K,_,_,Cout = B.shape

        Ns, Hs, Ws, Cs = A.strides
        Hss = Hs * self.stride
        Wss = Ws * self.stride

        Hout = (H - K) // self.stride + 1
        Wout = (W - K) // self.stride + 1

        outer_dim = N * Hout * Wout
        inner_dim = K*K*Cin

        Z = A.as_strided(shape=(N, Hout, Wout, K, K, Cin), strides=(Ns, Hss, Wss, Hs, Ws, Cs))
        Z = Z.compact().reshape((outer_dim, inner_dim))
        out = Z @ B.compact().reshape((inner_dim, Cout))
        return out.reshape((N, Hout, Wout, Cout))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, kernel = node.inputs
        N,H,W,Cin = X.shape
        K,_,_,Cout = kernel.shape
        print('strides=', self.stride, 'padding=', self.padding)
        print('out shape', out_grad.shape, 'Xshape', X.shape, 'kernel shape', kernel.shape)
        
        OG = dilate(out_grad, (1,2), self.stride-1) 
        print('OG shape', OG.shape)

        kf = transpose(flip(kernel, (0,1)))
        xgrad = conv(OG, kf, padding=K-1-self.padding)  # (H + 2P - K + 1) + 2(K-1-P) - (K-1)
        print('xgrad shape', xgrad.shape)

        XT = transpose(X, axes=(0,3))    # CinHWN
        OGT = transpose(transpose(OG, axes=(0,1)), axes=(1,2)) #HWNCout
        WG = conv(XT, OGT, padding=self.padding)  # CinKKCout
        print('XT shape', XT.shape, 'OGT shape', OGT.shape, 'WG shape', WG.shape)
        wgrad = transpose(transpose(WG, axes=(0,1)), axes=(1,2))

        return xgrad, wgrad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



