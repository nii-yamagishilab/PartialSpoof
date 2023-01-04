
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

#https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/e987ac603fa1accd386820a985a6dc2fd92dec5b/depreciate/pcgrad_ori.py

class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        input:
        - objectives: a list of objectives
        '''

        grads, shapes = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, shapes=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            # [tensor([ 0.0077,  0.0129, -0.0075,  ..., -0.0023, -0.0004,  0.0006 ], device='cuda:0'),
            #  tensor([ 0.0064,  0.0067, -0.0055,  ..., -0.0021, -0.0003,  0.0006 ],device='cuda:0')]
            # # after shuffle
            # [tensor([ 0.0064,  0.0067, -0.0055,  ..., -0.0021, -0.0003,  0.0006 ],device='cuda:0'),
            #  tensor([ 0.0077,  0.0129, -0.0075,  ..., -0.0023, -0.0004,  0.0006 ],device='cuda:0')]
            
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        pc_grad = torch.stack(pc_grad).mean(dim=0)
        return pc_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes = [], []
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network.

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        '''

        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
        return grad, shape
