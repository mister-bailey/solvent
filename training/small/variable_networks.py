import e3nn
from e3nn.networks import *
from e3nn.batchnorm import BatchNorm
#from radial import GaussianRadialModel

class VariableParityNetwork(torch.nn.Module):
    def __init__(self, Rs_in, muls, Rs_out, lmaxes, #layers=3,
                 max_radius=1.0, number_of_basis=3, radial_layers=3, radial_h=100,
                 feature_product=False, kernel=Kernel, convolution=Convolution,
                 radial_model=None, batch_norm=False, deep_batch_norm=False):
        super().__init__()

        self.batch_norm = batch_norm
        if radial_model is None:
            R = partial(GaussianRadialModel, max_radius=max_radius,
                        number_of_basis=number_of_basis, h=radial_h,
                        L=radial_layers, act=swish)#, batch_norm=deep_batch_norm)
        else:
            R = partial(radial_model, max_radius=max_radius, h=radial_h,
                        L=radial_layers, act=swish)

        modules = []

        Rs = Rs_in
        if batch_norm:
            modules.append(BatchNorm([(m,2 * l + 1) for (m,l,_) in Rs_in]))

        for mul, lmax in zip(muls, lmaxes):
            scalars = [(mul, l, p) for mul, l, p in [(mul, 0, +1), (mul, 0, -1)] if rs.haslinearpath(Rs, l, p)]
            act_scalars = [(mul, swish if p == 1 else tanh) for mul, l, p in scalars]

            nonscalars = [(mul, l, p) for l in range(1, lmax + 1) for p in [+1, -1] if rs.haslinearpath(Rs, l, p)]
            gates = [(rs.mul_dim(nonscalars), 0, +1)]
            act_gates = [(-1, sigmoid)]

            K = partial(kernel, RadialModel=R, selection_rule=partial(o3.selection_rule_in_out_sh, lmax=lmax))

            act = GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)
            conv = convolution(K(Rs, act.Rs_in))
            if batch_norm:
                #print(f"Rs = {Rs}")
                #print(f"act.Rs_in = {act.Rs_in}")
                bn = BatchNorm([(m,2 * l + 1) for (m,l,_) in act.Rs_in])

            if feature_product:
                tr1 = rs.TransposeToMulL(act.Rs_out)
                lts = LearnableTensorSquare(tr1.Rs_out, [(1, l, p) for l in range(lmax + 1) for p in [-1, 1]], allow_change_output=True)
                tr2 = torch.nn.Flatten(2)
                act = torch.nn.Sequential(act, tr1, lts, tr2)
                Rs = tr1.mul * lts.Rs_out
            else:
                Rs = act.Rs_out

            if batch_norm:
                block = torch.nn.ModuleList([conv, bn, act])
            else:
                block = torch.nn.ModuleList([conv, act])
            modules.append(block)

        self.layers = torch.nn.ModuleList(modules)

        K = partial(K, allow_unused_inputs=True)
        self.layers.append(convolution(K(Rs, Rs_out)))
        self.feature_product = feature_product

    def forward(self, input, *args, **kwargs):
        output = input
        N = args[0].shape[-2]
        if 'n_norm' not in kwargs:
            kwargs['n_norm'] = N

        if self.batch_norm:
            output = self.layers[0](output)
            for (i,(conv, bn, act)) in enumerate(self.layers[1:-1]):
                output = conv(output, *args, **kwargs)
                output = bn(output)
                output = act(output)
        else:
            for conv, act in self.layers[:-1]:
                output = conv(output, *args, **kwargs)
                output = act(output)

        layer = self.layers[-1]
        output = layer(output, *args, **kwargs)
        return output

