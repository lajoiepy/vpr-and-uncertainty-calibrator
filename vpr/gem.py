import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GemLayer(nn.Module):
    """ GeM layer implementation
        based on https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
    """    
    def __init__(self, p=3, mp=1, eps=1e-6, dim=512, var_dim=0, is_variance=False):
        super(GemLayer,self).__init__()
        self.p = Parameter(torch.ones(mp)*p)
        self.mp = mp
        self.eps = eps
        self.dim = dim
        self.is_variance = is_variance
        if self.is_variance:
            self.fc1 = nn.Linear(self.dim, self.dim)
            self.fc2 = nn.Linear(self.dim, var_dim)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()
            self.softplus = nn.Softplus()
        else:
            self.fc = nn.Linear(self.dim, self.dim-var_dim)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def forward(self, x):
        if self.is_variance:
            x = self.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps).squeeze()
            x = self.fc1(x)
            x = self.fc2(x)
            #x = self.relu(x)
            #x = self.sigmoid(x)
            x = self.softplus(x)
            return x
        else:
            x = self.gem(x, p=self.p, eps=self.eps).squeeze()
            x = self.fc(x)
            x = F.normalize(x, p=2, dim=0)
            return x
