import numpy as np
import h5py
from collections.abc import Sequence

def mul_greater(c, x):
    return (x // c + 1) * c

class Array:
    """
    Expandable array, based on an underlying numpy array
    Append increases length by 1.
    If check_bounds == False, indexing beyond the end (either
    setting or getting) also increases length, filling with
    zeros.
    """
    default_chunk_size = 1000
    
    def __init__(self, array=None, shape=(0,), chunk_size=None, **kwargs):
        if array is not None:
            array = np.array(array)
            shape = array.shape
        self.cross_shape = shape[1:]
        self.length = shape[0]
        if chunk_size is None:
            chunk_size = Array.default_chunk_size
        self.chunk_size = chunk_size
        if array is None:
            self.data = np.zeros((mul_greater(chunk_size, self.length), *self.cross_shape), **kwargs)
        else:
            self.data = np.resize(array, (mul_greater(chunk_size, self.length), *self.cross_shape))
            
    def __repr__(self):
        return repr(self.array)
        
    def __len__(self):
        return self.length
    
    @property
    def array(self):
        return self.data[:self.length]
    
    @array.setter
    def array(self, value):
        self.data[:self.length] = value
        
    @property
    def capacity(self):
        return len(self.data)
        
    @property
    def shape(self):
        return (self.length, *self.cross_shape)
    
    def __getitem__(self, key):
        return (self.array)[key]
        
    def __setitem__(self, key, value):
        self.array[key] = value
        
    def __iter__(self):
        return iter(self.array)
        
    def resize(self, new_length, shrink_data=False):
        if new_length < self.length:
            if shrink_data:
                self.data = np.resize(self.data, (mul_greater(self.chunk_size, new_length), *self.cross_shape))
                self.data[new_length:min(self.length, len(self.data))] = 0
            else:
                self.data[new_length:self.length] = 0
        elif new_length > len(self.data):
            self.data = np.resize(self.data, (mul_greater(self.chunk_size, new_length), *self.cross_shape))
        self.length = new_length
             
    def append(self, value):
        self.resize(self.length + 1)
        self.data[self.length - 1] = value    
        
    def __add__(self, other): return self.array + other
    def __sub__(self, other): return self.array - other
    def __mul__(self, other): return self.array * other
    def __matmul__(self, other): return self.array @ other
    def __truediv__(self, other): return self.array / other
    def __floordiv__(self, other): return self.array // other
    def __mod__(self, other): return self.array % other
    def __divmod__(self, other): return divmod(self.array, other)
    def __pow__(self, other): return self.array ** other
    def __lshift__(self, other): return self.array << other
    def __rshift__(self, other): return self.array >> other
    def __and__(self, other): return self.array & other
    def __xor__(self, other): return self.array ^ other
    def __or__(self, other): return self.array | other

    def __radd__(self, other): return other + self.array
    def __rsub__(self, other): return other - self.array
    def __rmul__(self, other): return other * self.array
    def __rmatmul__(self, other): return other @ self.array
    def __rtruediv__(self, other): return other / self.array
    def __rfloordiv__(self, other): return other // self.array
    def __rmod__(self, other): return other % self.array
    def __rdivmod__(self, other): return divmod(other, self.array)
    def __rpow__(self, other): return other**self.array
    def __rlshift__(self, other): return other << self.array
    def __rrshift__(self, other): return other >> self.array
    def __rand__(self, other): return other & self.array
    def __rxor__(self, other): return other ^ self.array
    def __ror__(self, other): return other | self.array
        
    def __iadd__(self, other):
        self.array += other
        return self
    def __isub__(self, other):
        self.array -= other
        return self
    def __imul__(self, other): 
        self.array *= other
        return self
    def __imatmul__(self, other): 
        self.array @= other
        return self
    def __itruediv__(self, other): 
        self.array /= other
        return self
    def __ifloordiv__(self, other): 
        self.array //= other
        return self
    def __imod__(self, other): 
        self.array %= other
        return self
    def __ipow__(self, other): 
        self.array ** other
        return self
    def __ilshift__(self, other): 
        self.array << other
        return self
    def __irshift__(self, other): 
        self.array >> other
        return self
    def __iand__(self, other): 
        self.array & other
        return self
    def __ixor__(self, other): 
        self.array ^ other
        return self
    def __ior__(self, other): 
        self.array | other
        return self
    

class H5Array(h5py.Dataset):
    default_chunk_size = 256
    default_compression = 'gzip'
            
    def __init__(self, h5, name, arg1=None, chunk_size=None, **kwargs):
        data = None
        if isinstance(arg1, tuple):
            shape = arg1
        elif isinstance(arg1, int):
            shape = (arg1,)
        elif arg1 is not None:
            data = np.array(arg1)
            shape = data.shape
        elif 'data' in kwargs:
            data = np.array(kwargs['data'])
            shape = data.shape
        elif 'shape' in kwargs:
            shape = kwargs['shape']
        elif h5 is not None and name in h5:
            shape = h5[name].shape
            chunk_size = h5[name].chunks[0]
        else:
            shape = (0,)
        
        if chunk_size is None:
            chunk_size = H5Array.default_chunk_size
        self.cross_shape = shape[1:]
        kwargs['shape'] = shape
        kwargs['maxshape'] = (None, *self.cross_shape)
        kwargs['chunks'] = (chunk_size, *self.cross_shape)
        if 'compression' not in kwargs:
            kwargs['compression'] = H5Array.default_compression
                
        if data is None:
            if 'dtype' not in kwargs:
                kwargs['dtype'] = h5[name].dtype
            dset = h5.require_dataset(name, **kwargs)
        else:
            if 'dtype' not in kwargs:
                kwargs['dtype'] = data.dtype
            dset = h5.create_dataset(name, **kwargs)
                
        super().__init__(dset.id)
        
    def append(self, value):
        super().resize(len(self)+1, axis=0)
        self[-1] = value
        
    def resize(self, size, axis=0):
        super().resize(size, axis)
        
    def resize_cross(self, size, axis=None):
        if axis is not None:
            super().resize(size, axis+1)
        elif isinstance(size, Sequence):
            assert len(self.size) == len(size) + 1
            super().resize((self.size[0], *size))
        else:
            super().resize(size, axis=1)

    
    