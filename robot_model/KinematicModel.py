from .KinematicTree import KinematicTree
from .model import OffsetModel
import torch
import numpy as np
from .utils import LinkNode, find_mother

class KinematicModel:
  def __init__(self, nJoint = 8, udim = 2):
    self.nJoint = nJoint
    self.udim = udim
    self.model = OffsetModel(udim=udim, nJoint=nJoint)
    self.kt = KinematicTree(nJoint)

  def predict(self, train_control, eval=False):
    nJoint = self.nJoint
    kt = self.kt
    
    traj=torch.empty([0,3])
    pjoints=torch.empty([0,nJoint,3])
    qs, rs = self.model(train_control)
    for i in range(len(qs)):
        for j in range(kt.N-1):
            kt.links[j+1] = LinkNode(id=j+1, name='link'+str(j+1), children=[j+2], b=rs[i][j]*kt.b[j].T,  a=kt.rpy[j], q=qs[i][j])
        kt.links[kt.N] = LinkNode(id=kt.N, name='link'+str(kt.N), children=[0], b=rs[i][kt.N-1]*kt.b[kt.N-1].T,  a=kt.rpy[j], q=qs[i][kt.N-1])

        kt.tree = find_mother(kt.links, 1)
        kt.tree[1].p = kt.root_positions.T
        kt.tree[1].R = torch.FloatTensor(np.eye(3))

        kt.forward_kinematics(1)
        pjoint = torch.empty([0, 3])
        for j in range(nJoint):
            pjoint = torch.cat((pjoint, torch.reshape(kt.tree[j+1].p, (-1, 3))), dim=0)
        pjoints = torch.cat((pjoints, torch.reshape(pjoint, (-1, nJoint, 3))), dim=0)
        traj = torch.cat((traj, torch.reshape(kt.tree[nJoint].p, (-1, 3))), dim=0)

    return traj, pjoints