# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name, line-too-long
import torch
import torch.nn as nn

# Changed behaviour from original e3nn.batchnorm:
#   running_var and running_mean are now used both for testing and training

class BatchNorm(nn.Module):
    def __init__(self, Rs, eps=1e-5, momentum=0.1, affine=True, reduce='mean', normalization='component'):
        '''
        Batch normalization layer for orthonormal representations
        It normalizes by the norm of the representations.
        Not that the norm is invariant only for orthonormal representations.
        Irreducible representations `o3.irr_repr` are orthonormal.

        input shape : [batch, [spacial dimensions], stacked orthonormal representations]

        :param Rs: list of tuple (multiplicity, dimension)
        :param eps: avoid division by zero when we normalize by the variance
        :param momentum: momentum of the running average
        :param affine: do we have weight and bias parameters
        :param reduce: method to contract over the spacial dimensions
        '''
        super().__init__()

        self.Rs = [(m, d) for m, d in Rs if m * d > 0]
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        num_scalar = sum(m for m, d in Rs if d == 1)
        num_features = sum(m for m, d in Rs)

        self.register_buffer('running_mean', torch.zeros(num_scalar))
        self.register_buffer('running_var', torch.ones(num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return "{} (Rs={}, eps={}, momentum={})".format(
            self.__class__.__name__,
            self.Rs,
            self.eps,
            self.momentum)

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input):  # pylint: disable=redefined-builtin
        '''
        :param input: [batch, ..., stacked features]
        '''
        *size, dim = input.shape
        input = input.reshape(-1, dim)  # [sample, stacked features]

        if self.training:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for m, d in self.Rs:
            field = input[:, ix: ix + m * d]  # [sample, mul * repr]
            ix += m * d

            # [sample, mul, repr]
            field = field.reshape(-1, m, d)

            if d == 1:  # scalars
                if self.training:
                    field_mean = field.detach().mean(0).reshape(m)  # [mul]
                    field_mean = self._roll_avg(self.running_mean[irm:irm + m], field_mean)
                    new_means.append(field_mean)
                else:
                    field_mean = self.running_mean[irm: irm + m]
                irm += m

                # [sample, mul, repr]
                field = field - field_mean.reshape(m, 1)

            if self.training:
                if self.normalization == 'norm':
                    #field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                    field_norm = field.detach().pow(2).sum(-1)  # [sample, mul]
                elif self.normalization == 'component':
                    #field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                    field_norm = field.detach().pow(2).mean(-1)  # [sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == 'mean':
                    #field_norm = field_norm.mean(1)  # [batch, mul]
                    field_norm = field_norm.mean(0)  # [mul]
                elif self.reduce == 'max':
                    #field_norm = field_norm.max(1).values  # [batch, mul]
                    field_norm = field_norm.max(0).values  # [..., mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                while field_norm.dim() > 1:
                    field_norm = field_norm.mean(0)  # [mul]
                field_norm = self._roll_avg(self.running_var[irv: irv + m], field_norm)
                new_vars.append(field_norm)
            else:
                field_norm = self.running_var[irv: irv + m]
            irv += m

            field_norm_inv = (field_norm + self.eps).pow(-0.5)  # [mul]

            if self.affine:
                weight = self.weight[iw: iw + m]  # [mul]
                iw += m

                field_norm = field_norm * weight  # [mul]

            field = field * field_norm_inv.reshape(m, 1)  # [sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + m]  # [mul]
                ib += m
                field += bias.reshape(m, 1)  # [sample, mul, repr]

            #fields.append(field.reshape(batch, -1, m * d))  # [batch, sample, mul * repr]
            fields.append(field.reshape(-1, m * d))  # [sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        if self.training:
            assert irm == self.running_mean.numel()
            assert irv == self.running_var.size(0)
        if self.affine:
            assert iw == self.weight.size(0)
            assert ib == self.bias.numel()

        if self.training:
            self.running_mean.copy_(torch.cat(new_means))
            self.running_var.copy_(torch.cat(new_vars))

        output = torch.cat(fields, dim=-1)  # [sample, stacked features]
        return output.reshape(*size, dim)
