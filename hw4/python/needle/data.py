import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            H, W, C = img.shape
            x = img.reshape(1, H*W, C)
            x180 = x[0][::-1].reshape(H, W, C)
            return x180[::-1]
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        data = np.pad(img, ((self.padding, self.padding),(self.padding,self.padding),(0,0)), 'constant', constant_values=0)
        H, W, C = img.shape
        x = self.padding + shift_x
        y = self.padding + shift_y
        return np.copy(data[x: x + H, y: y + W])
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.idx = 0
        if self.shuffle:
            index = np.arange(len(self.dataset))
            np.random.shuffle(index)
            self.ordering = np.array_split(index, range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.idx >= len(self.ordering):
            raise StopIteration
        indices = self.ordering[self.idx]
        self.idx += 1
        return [Tensor(x, device=None, requires_grad=False) for x in self.dataset[indices]]
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION     
        self.images, self.labels = self.parse_mnist(image_filename, label_filename)
        print(len(self.images), len(self.labels), self.images.shape, self.labels.shape)
        super().__init__(transforms)
        ### END YOUR SOLUTION

    def parse_mnist(self, image_filename, label_filename):
        ### BEGIN YOUR SOLUTION
        import gzip
        import struct
        with gzip.open(label_filename, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))

        with gzip.open(image_filename, 'rb') as f:
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).astype(np.float32)
            maxval = np.max(images)
            minval = np.min(images)
            images = (minval + (images.reshape((size, rows*cols)) - minval) / (maxval - minval))
            print('parse_mnist', size, rows, cols)
        return (images.reshape(size, rows, cols, 1), labels)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.apply_transforms(self.images[index])
        return (X.reshape(X.shape[0], -1), self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            self.X, self.y = self._load_cifar10(base_folder)
        else:
            filename = os.path.join(base_folder, 'test_batch')
            self.X, self.y = self._load_cifar_batch(filename)

        self.X /= 255
        print(self.X.shape, self.X.dtype, self.y.shape, self.y.dtype)
        super().__init__(transforms)
        ### END YOUR SOLUTION

    def _load_cifar_batch(self, filename):
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')  
            X = dict[b'data']
            Y = dict[b'labels']
            X = X.reshape(10000, 3, 32, 32)
            return X.astype('float32'), np.array(Y)

    def _load_cifar10(self, root):
        xs = []
        ys = []
        for b in range(1, 6):
            filename = os.path.join(root, 'data_batch_%d' % b)
            X, Y = self._load_cifar_batch(filename)
            xs.append(X)
            ys.append(Y)
        return np.concatenate(xs), np.concatenate(ys)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X = self.apply_transforms(self.X[index])
        return (X, self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        with open(path) as f:
            cnt = 0
            idxs_list = []
            for line in f:
                tokens = line.strip().split(' ')
                tokens.append('<eos>')
                idxs = [self.dictionary.add_word(word) for word in tokens]
                idxs_list += idxs
                cnt += 1
                if max_lines is not None and cnt >= max_lines:
                    break
            print('tokens:', len(idxs_list))
            return idxs_list
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    print('batchify data:', len(data), batch_size)
    print('top100 data:', data[:100])
    nbatch = len(data) // batch_size
    return np.array(data[:nbatch * batch_size]).reshape(batch_size, nbatch).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    bs = batches.shape[1]
    sq = bptt #min(bptt, batches.shape[0] - i - 1)
    print('get batch i=', i, 'from', batches.shape, 'by', sq)
    x = batches[i:i+sq, :]
    y = batches[i+1:i+sq+1, :].reshape(sq*bs, )
    return Tensor(x, device=device, dtype=dtype), Tensor(y, device=device, dtype=dtype)
    ### END YOUR SOLUTION