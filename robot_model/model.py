import torch.nn as nn

##### Model construction #####
def mlp(udim, hdims, actv, output_actv):
    layers = []
    prev_hdim = udim
    for hdim in hdims[:-1]:
        layers.append(nn.Linear(prev_hdim, hdim, bias=True))
        layers.append(actv)
        prev_hdim = hdim
    layers.append(nn.Linear(prev_hdim, hdims[-1]))
    if output_actv is None:
        return nn.Sequential(*layers)
    else:
        layers.append(output_actv)
        return nn.Sequential(*layers)

class OffsetModel(nn.Module):
    def __init__(self, nJoint, udim=2, hdims=[256,256], actv=nn.ReLU(), output_actv=nn.ReLU()):
        super(OffsetModel, self).__init__()
        self.output_actv = output_actv
        self.qnet = mlp(udim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.rnet = mlp(udim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.q = nn.Linear(in_features=hdims[-1], out_features=(nJoint))
        self.r = nn.Linear(in_features=hdims[-1], out_features=(nJoint))
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.qnet(x)
        q = self.q(output)
        output = self.rnet(x)
        r = self.r(output)
        r = self.relu(r)
        return q, r