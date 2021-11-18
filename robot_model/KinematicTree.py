import torch
from .utils import rpy2rot, LinkNode, rodrigues_torch, rodrigues, find_mother
print(torch.__version__)

class KinematicTree:
    """Custom Model Class"""

    def __init__(self, N, tree_name='kinematic_tree', device='cpu'):
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
        self.rpy = torch.rand(self.N, 3, requires_grad=True)
        self.b = torch.rand(self.N, 3, requires_grad=True)

        # root position
        self.root_positions = torch.Tensor([0, 0, 0.2])

        self.device = device

    def forward_kinematics(self, node_id):
        if node_id == 0: # For end of the kinematic chain. (NULL)
            return None
        if node_id != 1: # If node is not body
            mother_id = self.tree[node_id].mother
            self.tree[node_id].p = self.tree[mother_id].R @ self.tree[node_id].b + self.tree[mother_id].p
            self.tree[node_id].R = self.tree[mother_id].R @ rodrigues_torch(self.tree[node_id].a, self.tree[node_id].q)

        for child_id in self.tree[node_id].children:
            self.forward_kinematics(child_id)