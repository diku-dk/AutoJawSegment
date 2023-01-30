import torch
device = 'cpu'
I = torch.eye(4, device=device)
P = torch.zeros((4,4), device=device)
R = P + I #affine matrix

import numpy as np

# def affine(pts,R,device):
#     tpst=torch.cat((pts,torch.ones((pts.shape[0],1),device=device,dtype=pts.dtype)),1)
#     tpts=R.mm(tpst.t()).t()
#     return tpts[:,:3]


def affine(points,mat,device):
    points_new=torch.cat((pts,torch.ones((pts.shape[0],1),device=device,dtype=pts.dtype)),1)
    points_new=mat.mm(tpst.t()).t()
    return points_new[:,:3]

def create_loss():
    return loss

def main():
    device='cuda:0'
    I=torch.eye(4,device=device)
    P=torch.zeros((4,4),device=device)
    P.requires_grad=True
    for i in range(100):
        print(i)
        lr = 0.01
        mat = P + I
        loss=create_loss()
        loss.backward()
        loss_old=loss.data
        P_test = P.data - lr * P.grad.data
        mat_test = P_test + I
        loss_new=create_loss()
        while loss_new > loss_old:
            lr = lr * 0.5
            P_test = P.data - lr * P.grad.data
            mat_test = P_test + I
            loss_new = create_loss()
        P.data.sub_(lr * P.grad.data)
        P.grad.data.zero_()