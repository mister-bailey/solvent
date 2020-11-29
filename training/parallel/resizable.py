import numpy as np

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
    
    def __init__(self, array=None, shape=(0,), chunk_size=None, check_bounds=True, **kwargs):
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
    
