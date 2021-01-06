# Customized implementation of SGD optimizer.

import torch
from torch.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if isinstance(group['lr'], torch.Tensor):
                    try:
                        p.data = p.data + torch.mul(-group['lr'].data, d_p)
                    except RuntimeError: # Hack for group-channel-wise training
                        nr_channels = d_p.size()[1]
                        apply_channels = nr_channels//4
                        ten = []
                        for idx, i in enumerate(range(0, nr_channels, apply_channels)):
                            ten.append(torch.mul(-group['lr'].data[:,idx:idx+1,:,:], d_p[:,i:i+apply_channels,:,:]))
                        fin = torch.cat(ten, dim=1)
                        p.data = p.data + fin
                else:
                    p.data.add_(-group['lr'], d_p)

####################################### TEST ###################################
def test_optimizer():
    tensor = torch.ones((1,3,3,3))
    lr = torch.ones((1,3,3,3))
    param_dict = {
        'params': tensor,
        'lr': lr
    }
    optim = SGD(param_dict)
    # optim.zero_grad()
    optim.step()
    print(tensor)

if __name__ == "__main__":
    test_optimizer()