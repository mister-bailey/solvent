import math
import torch
from e3nn.radial import FC

class LaurentPolynomial(torch.nn.Module):
    def __init__(self, out_dim, max_radius, min_degree, max_degree, h, L, act, min_radius=None):
        assert min_degree <= 0
        assert max_degree >= 0
        assert max_degree - min_degree >= 1

        super().__init__()

        powers = torch.linspace(min_degree, max_degree, max_degree - min_degree + 1).unsqueeze(0)
        self.register_buffer('powers', powers)
        if min_radius is None:
            min_radius = max_radius / 1000
        cp = torch.reciprocal(torch.pow(max_radius,powers+1))
        cn = torch.reciprocal(torch.pow(min_radius,powers+1))
        c = torch.min(cp,cn)
        #print(c)
        self.register_buffer('c', c)

        self.max_radius = max_radius
        self.min_radius = min_radius
        #self.epsilon = epsilon

        self.f = FC(max_degree - min_degree + 1, out_dim, h=h, L=L, act=act)

    def forward(self, x):
        x = torch.clamp(x, self.min_radius, self.max_radius)
        x = x.unsqueeze(1).pow(self.powers) * self.c
        #x = x.unsqueeze(1).pow(self.powers)
        return self.f(x)

