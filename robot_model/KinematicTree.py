import torch
import numpy as np
from torch._C import dtype
from utils import rpy2rot, LinkNode, rodrigues, find_mother
from model import OffsetModel
print(torch.__version__)

class KinematicTree:
    """Custom Model Class"""

    def __init__(self, N, q, r, tree_name='kinematic_tree', device='cpu'):
        """
        Args:
            N: the num of joint
            q: joint angles
            r: link length scales
            root_name (str, optional): name of root joint. Defaults to 'Hips'.
            device (str, optional): device to use. Defaults to 'cpu'.
        """
        self.root_name = tree_name
        self.N = N # num of joint

        self.links = {}
        self.links[0] = LinkNode(id=0,  name='NULL')
        self.roll = np.pi*torch.rand(self.N, requires_grad=True)
        self.pitch = np.pi*torch.rand(self.N, requires_grad=True)
        self.yaw = np.pi*torch.rand(self.N, requires_grad=True)
        self.b = torch.rand(self.N, 3, requires_grad=True)

        for i in range(N-1):
            # a = rpy2rot(self.roll[i], self.pitch[i], self.yaw[i])
            self.links[i+1] = LinkNode(id=i+1,  name='link'+str(i+1), children=[i+2], b=r[i]*self.b[i].T,  a=[self.roll[i].detach().numpy(), self.pitch[i].detach().numpy(), self.yaw[i].detach().numpy()], q=q[i])

        self.links[N] = LinkNode(id=N, name='link'+str(N), children=[0], b=r[N-1]*self.b[N-1].T,  a=[self.roll[N-1].detach().numpy(), self.pitch[N-1].detach().numpy(), self.yaw[N-1].detach().numpy()], q=q[N-1])
        
        self.tree = find_mother(self.links, 1)

        # root position
        root_positions = torch.randn(3)
        self.tree[1].p = root_positions.T
        self.tree[1].R = torch.FloatTensor(np.eye(3))
        self.device = device

    def forward_kinematics(self, node_id):
        if node_id == 0: # For end of the kinematic chain. (NULL)
            return None
        if node_id != 1: # If node is not body
            mother_id = self.tree[node_id].mother
            self.tree[node_id].p = (self.tree[mother_id].R @ self.tree[node_id].b + self.tree[mother_id].p)
            self.tree[node_id].R = self.tree[mother_id].R @ torch.FloatTensor(rodrigues(self.tree[node_id].a, self.tree[node_id].q.detach().numpy()))

        for child_id in self.tree[node_id].children:
            self.forward_kinematics(child_id)

nJoint = 10
model = OffsetModel(udim=1, nJoint=nJoint)
x=torch.tensor([0.1])
q, r = model(x)
kt = KinematicTree(nJoint, q, r)
kt.forward_kinematics(1)