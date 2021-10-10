from matplotlib import animation
import torch
import torch.nn.functional as F
import numpy as np
from utils import rpy2rot, LinkNode, rodrigues_torch, rodrigues, find_mother
from model import OffsetModel
from scipy.special import fresnel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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

def update():
    return None

nJoint = 8
interpolate = 25
model = OffsetModel(udim=1, nJoint=nJoint)
u = np.linspace(0, 2, interpolate)

x = fresnel(u)[0]
y = fresnel(u)[1]
x = torch.FloatTensor(x.reshape(-1, 1))

kt = KinematicTree(nJoint)

params = list(model.parameters())+list([kt.rpy])+list([kt.b])
optimizer = torch.optim.Adam(params)
target = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), np.zeros(shape=(interpolate,1))), axis=1)
target = torch.FloatTensor(target)

for epoch in range(100000000):
    traj=torch.empty([0,3])
    pjoints=torch.empty([0,nJoint,3])

    input = torch.Tensor(u.reshape(-1, 1))
    qs, rs = model(input)
    for i in range(interpolate):
        for j in range(kt.N-1):
            # a = rpy2rot(self.roll[i], self.pitch[i], self.yaw[i])
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
    
    loss = F.mse_loss(traj.squeeze(), target.squeeze(), reduction='sum')

    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print('hey', list(model.qnet.parameters())[0].grad)
    # print('hey', list(model.rnet.parameters())[0].grad)
    # print('hey', kt.rpy.grad)
    # print('hey', kt.b.grad)
    if(epoch % 100 == 0):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def update(frame, data, data2, line, line2):
            line.set_data(data[:2, :frame])
            line.set_3d_properties(data[2, :frame])
            line2.set_data(data2[frame, 0:nJoint, 0], data2[frame, 0:nJoint, 1])
            line2.set_3d_properties(data2[frame, 0:nJoint, 2])
            return line
        
        data = traj.detach().numpy().T
        data2 = pjoints.detach().numpy()
        line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'bo')
        line2, = ax.plot(data2[0, 0:nJoint, 0], data2[0, 0:nJoint, 1], data2[0, 0:nJoint, 2], '.-')

        ax.set_xlim3d([-1.0, 1.0])
        ax.set_ylim3d([-1.0, 1.0])
        ax.set_zlim3d([-1.0, 1.0])

        plt.plot(target[:,0], target[:,1], target[:,2], c='k')
        ani = FuncAnimation(fig, update, fargs=[data, data2, line, line2], frames=range(interpolate))
        # plt.plot(traj.detach().numpy()[t,0], traj.detach().numpy()[t,1], traj.detach().numpy()[t,2])
        plt.show()

        
