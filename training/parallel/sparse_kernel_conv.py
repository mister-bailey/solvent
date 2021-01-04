import torch

from e3nn import o3, rs, rsh
from e3nn.kernel import Kernel

class SparseKernelConv(Kernel):

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1, custom_backward=False):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_edges] ~ [a, b]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_edges, 3]
                       edge_r = position_target - position_source
        :param size: n_points or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_points, dim(Rs_out)]
        """
        assert edge_r.shape[1] == 3

        radii = edge_r.norm(2, dim=-1) 

        # precompute all needed spherical harmonics
        y = rsh.spherical_harmonics_xyz(self.set_of_l_filters, edge_r)  # [batch, a, b, l_filter * m_filter]

        y[radii == 0] = 0

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        r = self.R(radii.flatten()).reshape(*radii.shape, -1)  # [*_, n_edges, l_out * l_in * mul_out * mul_in * l_filter]
        r = r.clone()
        r[radii == 0] = 0

        if custom_backward:
            assert False, "Custom backward for sparse kernel: not coded yet!"
            #output = KernelConvFn.apply(
            #    features, edge_index, y, r, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters
            #)
        else:
            output = kernel_conv_fn_forward(
                features, edge_index, y, r, self.norm_coef, self.Rs_in, self.Rs_out, self.selection_rule, self.set_of_l_filters
            )
        
        output.div_(n_norm ** 0.5)

        # Case r > 0
        #if radii.shape[1] == radii.shape[2]:
        output += torch.einsum('ij,aj->ai', self.linear(), features)

        return output 

def kernel_conv_fn_forward(F, edge_index, Y, R, norm_coef, Rs_in, Rs_out, selection_rule, set_of_l_filters):
    """
    :param F: tensor [b, l_in * mul_in * m_in]
    :param Y: tensor [n_edges, l_filter * m_filter]
    :param R: tensor [n_edges, l_out * l_in * mul_out * mul_in * l_filter]
    :param norm_coef: tensor [l_out, l_in]
    :return: tensor [a, l_out * mul_out * m_out, l_in * mul_in * m_in]
    """
    n_edges = Y.shape[-2]
    n_atoms = F.shape[-2]
    n_out = rs.dim(Rs_out)

    kernel_conv = Y.new_zeros(n_atoms, n_out)

    # note: for the normalization we assume that the variance of R[i] is one
    begin_R = 0

    begin_out = 0
    for i, (mul_out, l_out, p_out) in enumerate(Rs_out):
        s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))
        begin_out += mul_out * (2 * l_out + 1)

        begin_in = 0
        for j, (mul_in, l_in, p_in) in enumerate(Rs_in):
            s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))
            begin_in += mul_in * (2 * l_in + 1)

            l_filters = selection_rule(l_in, p_in, l_out, p_out)
            if not l_filters:
                continue

            # extract the subset of the `R` that corresponds to the couple (l_out, l_in)
            n = mul_out * mul_in * len(l_filters)
            sub_R = R[..., begin_R: begin_R + n].reshape(
                n_edges, mul_out, mul_in, -1
            )  # [n_edges, mul_out, mul_in, l_filter]
            begin_R += n

            K = 0
            for k, l_filter in enumerate(l_filters):
                offset = sum(2 * l + 1 for l in set_of_l_filters if l < l_filter)
                sub_Y = Y[..., offset: offset + 2 * l_filter + 1]  # [n_edges, m]

                C = o3.wigner_3j(l_out, l_in, l_filter, cached=True, like=kernel_conv)  # [m_out, m_in, m]

                # i - tensor product index for output
                # j - tensor product index for feature (SUMMED)
                # k - tensor product index for edge spherical harmonic Y (SUMMED)
                # u - multiplicity output index
                # v - multiplicity input index (SUMMED)
                # a - atom ~ edge[0]
                # b - atom ~ edge[1] (SUMMED SPARSELY)
                
                EF = F[edge_index[1], s_in].reshape(n_edges, mul_in, -1) # [num_edges, mul_in, J]
                D = norm_coef[i, j] * torch.einsum("ijk,ek,euv,evj->eui",
                        C, sub_Y, sub_R[..., k], EF) # [num_edges, mul_out, I]
                K += scatter_add(D, edge_index[0], 0, n_atoms) # [n_atoms, mul_out, I]

            if not isinstance(K, int):
                kernel_conv[..., s_out] += K.reshape(n_atoms, -1)

    return kernel_conv

class DummyConvolution(torch.nn.Module):
    def __init__(self, kernel, *args, **kwargs):
        super().__init__()
        self.kernel = kernel
    def forward(self, *args, **kwargs):
        return self.kernel.forward(*args, **kwargs)

def DummyConvolutionFn(kernel_conv):
    return kernel_conv

from typing import Optional

# scatter code adapted from pytorch-scatter
@torch.jit.script
def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim
    for _ in range(0, dim):
        index = index.unsqueeze(0)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    index = index.expand_as(src)
    if out is None:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        return out.scatter_add(dim, index, src)
    else:
        out.scatter_add_(dim, index, src)
        return out


