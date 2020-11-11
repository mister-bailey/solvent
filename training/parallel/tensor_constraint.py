import torch
#torch.set_default_dtype(torch.float64)
from e3nn.tensor.cartesian_tensor import CartesianTensor

Rs_out, Q = CartesianTensor(torch.eye(3)).to_irrep_transformation()

Rs_out = [(m,l,1) for m,l,_ in Rs_out] # ??????

conversion = torch.zeros(9,10)
conversion[0,0] = 1
conversion[1:9,1:10] = Q[1:9]

c_t = conversion.t()

def convert(data):
    global c_t
    c_t = c_t.to(data.device)
    return data @ c_t