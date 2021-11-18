import torch
import torch.nn.functional as F
import numpy as np
from robot_model.KinematicModel import KinematicModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random

files = [np.loadtxt('target/2dim_log_spiral_{}.txt'.format(i+1), delimiter = ' ', skiprows = 1, dtype = 'float') for i in range(1000)]
target_data = np.array(files)
nEval = 2
nBatch = 300
nTrain = 400
nJoint = 5
train_idx = random.sample(range(1000), nTrain)
eval_idx = list(range(1000))
km = KinematicModel(nJoint = 5, nBatch = nBatch, nTrain = nTrain)

# folding
for i in train_idx:
    eval_idx.remove(i)

# Select specific trajectory
# test1 = eval_idx.index(944)
# test2 = eval_idx.index(920)
control = target_data[train_idx,:,3:5].reshape(-1,2)
target_position = target_data[train_idx,:,0:3].reshape(-1,3)

params = list(km.model.parameters())+list([km.kt.rpy])+list([km.kt.b])
optimizer = torch.optim.Adam(params, weight_decay=0.01)
for epoch in range(100000000):
    randidx = random.sample(range(len(control)),nBatch)
    train_control = torch.Tensor(control[randidx])
    train_position = torch.Tensor(target_position[randidx])
  
    traj, pjoints = km.predict(train_control)
    
    loss = F.mse_loss(traj, train_position, reduction='mean')
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print('hey', list(model.qnet.parameters())[0].grad)
    # print('hey', list(model.rnet.parameters())[0].grad)
    # print('hey', kt.rpy.grad)
    # print('hey', kt.b.grad)

    # Eval
    if(epoch % 500 == 0):
        randidx = random.sample(eval_idx,nEval)
        for n in range(nEval):
            eval_train_control = torch.Tensor(target_data[randidx[n],:,3:5].reshape(-1,2))
            eval_train_position = torch.Tensor(target_data[randidx[n],:, 0:3].reshape(-1,3))
            eval_traj, eval_pjoints = km.predict(eval_train_control, eval=True)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            def update(frame, data, data2, line, line2):
                line.set_data(data[:2, :frame])
                line.set_3d_properties(data[2, :frame])
                line2.set_data(data2[frame, 0:nJoint, 0], data2[frame, 0:nJoint, 1])
                line2.set_3d_properties(data2[frame, 0:nJoint, 2])
                return line
            
            data = eval_traj.detach().numpy().T
            data2 = eval_pjoints.detach().numpy()
            line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'bo')
            line2, = ax.plot(data2[0, 0:nJoint, 0], data2[0, 0:nJoint, 1], data2[0, 0:nJoint, 2], '.-')

            ax.set_xlim3d([-50.0, 50.0])
            ax.set_ylim3d([-50.0, 50.0])
            ax.set_zlim3d([-50.0, 50.0])

            plt.plot(eval_train_position[:,0], eval_train_position[:,1], eval_train_position[:,2], c='k')
            ani = FuncAnimation(fig, update, fargs=[data, data2, line, line2], frames=range(299))
            # plt.plot(traj.detach().numpy()[t,0], traj.detach().numpy()[t,1], traj.detach().numpy()[t,2])
            loss = F.mse_loss(eval_traj, eval_train_position)
            print(epoch, loss)

            plt.show()