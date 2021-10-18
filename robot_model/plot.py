import torch
import numpy as np
import matplotlib.pyplot as plt
import random

files = [np.loadtxt('target/2dim_log_spiral_{}.txt'.format(i+1), delimiter = ' ', skiprows = 1, dtype = 'float') for i in range(1000)]
target_data = np.array(files)

nBatch = 300
nEval = 100
eval_idx = list(range(1000))
# Eval
randidx = random.sample(range(len(eval_idx)),nEval)
for n in range(nEval):
    eval_train_position = torch.Tensor(target_data[randidx[n],:, 0:3].reshape(-1,3))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d([-50.0, 50.0])
    ax.set_ylim3d([-50.0, 50.0])
    ax.set_zlim3d([-50.0, 50.0])

    plt.plot(eval_train_position[:,0], eval_train_position[:,1], eval_train_position[:,2], c='k')
    plt.show()
