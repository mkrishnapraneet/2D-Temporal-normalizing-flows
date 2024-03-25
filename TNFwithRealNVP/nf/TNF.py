# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:51:38 2020

@author: Yubin Lu
"""
import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    # def __init__(self, dim, hidden_dim = 32, base_network=FCNN):
    #     super().__init__()
    #     self.dim = dim

    #     self.t1 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s1 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.t2 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s2 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.t3 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s3 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.t4 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s4 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.t5 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s5 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.t6 = base_network(2 + dim // 2, dim // 2, hidden_dim)
    #     self.s6 = base_network(2 + dim // 2, dim // 2, hidden_dim)

    # def forward(self, x):            
    #     noChange, change = x[:,[0,1,3]], x[:,2].reshape(-1, 1)
    #     t1_transformed = self.t1(noChange)
    #     s1_transformed = self.s1(noChange)
    #     changed_new = t1_transformed + change * torch.exp(s1_transformed)
    #     z1 = torch.cat([noChange[:,0].reshape(-1, 1), noChange[:,1].reshape(-1, 1), changed_new, noChange[:,2].reshape(-1, 1)], dim=1)
       
    #     noChange1, change1 = z1[:,[0,2,3]], z1[:,1].reshape(-1, 1)
    #     t2_transformed = self.t2(noChange1)
    #     s2_transformed = self.s2(noChange1)
    #     changed_new1 = t2_transformed + change1 * torch.exp(s2_transformed)
    #     z2 = torch.cat([noChange1[:,0].reshape(-1, 1), changed_new1, noChange1[:,1].reshape(-1, 1), noChange1[:,2].reshape(-1, 1)], dim=1)
        
    #     noChange2, change2 = z2[:,[1,2,3]], z2[:,0].reshape(-1, 1)
    #     t3_transformed = self.t3(noChange2)
    #     s3_transformed = self.s3(noChange2)
    #     changed_new2 = t3_transformed + change2 * torch.exp(s3_transformed)
    #     z3 = torch.cat([changed_new2, noChange2], dim=1)
       
    #     noChange3, change3 = z3[:,[0,1,3]], z3[:,2].reshape(-1, 1)
    #     t4_transformed = self.t4(noChange3)
    #     s4_transformed = self.s4(noChange3)
    #     changed_new3 = t4_transformed + change3 * torch.exp(s4_transformed)
    #     z4 = torch.cat([noChange3[:,0].reshape(-1, 1), noChange3[:,1].reshape(-1, 1), changed_new3, noChange3[:,2].reshape(-1, 1)], dim=1)

    #     noChange4, change4 = z4[:,[0,2,3]], z4[:,1].reshape(-1, 1)
    #     t5_transformed = self.t5(noChange4)
    #     s5_transformed = self.s5(noChange4)
    #     changed_new4 = t5_transformed + change4 * torch.exp(s5_transformed)
    #     z5 = torch.cat([noChange4[:,0].reshape(-1, 1), changed_new4, noChange4[:,1].reshape(-1, 1), noChange4[:,2].reshape(-1, 1)], dim=1)

    #     noChange5, change5 = z5[:,[1,2,3]], z5[:,0].reshape(-1, 1)
    #     t6_transformed = self.t6(noChange5)
    #     s6_transformed = self.s6(noChange5)
    #     changed_new5 = t6_transformed + change5 * torch.exp(s6_transformed)
    #     z = torch.cat([changed_new5, noChange5], dim=1)

    #     # remove the last column of z
    #     z = z[:, :-1]

    #     log_det = torch.sum(s1_transformed + s2_transformed + s3_transformed + s4_transformed + s5_transformed + s6_transformed, dim=1)
    #     c1 = 1/(2*np.pi)
    #     c2 = -torch.sum(z**2, 1)/2
    #     pz = torch.mul(c1, torch.exp(c2))
    #     log_px = torch.log(pz) + log_det
    #     px = torch.exp(log_px)
    #     return z, log_det, px
    
    # def inverse(self, z):
    #     noChange5, change5 = z[:,[1,2,3]], z[:,0].reshape(-1, 1)
    #     t6_transformed = self.t6(noChange5)
    #     s6_transformed = self.s6(noChange5)
    #     changed_new5 = (change5 - t6_transformed) * torch.exp(-s6_transformed)
    #     x5 = torch.cat([changed_new5, noChange5], dim=1)

    #     noChange4, change4 = x5[:,[0,2,3]], x5[:,1].reshape(-1, 1)
    #     t5_transformed = self.t5(noChange4)
    #     s5_transformed = self.s5(noChange4)
    #     changed_new4 = (change4 - t5_transformed) * torch.exp(-s5_transformed)
    #     x4 = torch.cat([noChange4[:,0].reshape(-1, 1), changed_new4, noChange4[:,1].reshape(-1, 1), noChange4[:,2].reshape(-1, 1)], dim=1)

    #     noChange3, change3 = x4[:,[0,1,3]], x4[:,2].reshape(-1, 1)
    #     t4_transformed = self.t4(noChange3)
    #     s4_transformed = self.s4(noChange3)
    #     changed_new3 = (change3 - t4_transformed) * torch.exp(-s4_transformed)
    #     x3 = torch.cat([noChange3[:,0].reshape(-1, 1), noChange3[:,1].reshape(-1, 1), changed_new3, noChange3[:,2].reshape(-1, 1)], dim=1)

    #     noChange2, change2 = x3[:,[1,2,3]], x3[:,0].reshape(-1, 1)
    #     t3_transformed = self.t3(noChange2)
    #     s3_transformed = self.s3(noChange2)
    #     changed_new2 = (change2 - t3_transformed) * torch.exp(-s3_transformed)
    #     x2 = torch.cat([changed_new2, noChange2], dim=1)

    #     noChange1, change1 = x2[:,[0,2,3]], x2[:,1].reshape(-1, 1)
    #     t2_transformed = self.t2(noChange1)
    #     s2_transformed = self.s2(noChange1)
    #     changed_new1 = (change1 - t2_transformed) * torch.exp(-s2_transformed)
    #     x1 = torch.cat([noChange1[:,0].reshape(-1, 1), changed_new1, noChange1[:,1].reshape(-1, 1), noChange1[:,2].reshape(-1, 1)], dim=1)

    #     noChange, change = x1[:,[0,1,3]], x1[:,2].reshape(-1, 1)
    #     t1_transformed = self.t1(noChange)
    #     s1_transformed = self.s1(noChange)
    #     changed_new = (change - t1_transformed) * torch.exp(-s1_transformed)
    #     x = torch.cat([noChange[:,0].reshape(-1, 1), noChange[:,1].reshape(-1, 1), changed_new, noChange[:,2].reshape(-1, 1)], dim=1)

    #     log_det = torch.sum(-s1_transformed - s2_transformed - s3_transformed - s4_transformed - s5_transformed - s6_transformed, dim=1)
    #     return x, log_det


    def __init__(self, dim, hidden_dim = 32, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.t3 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s3 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.t4 = base_network(1 + dim // 2, dim // 2, hidden_dim)
        self.s4 = base_network(1 + dim // 2, dim // 2, hidden_dim)

    
    
    def forward(self, x):            
        lower, upper = x[:,0::2], x[:,1].reshape(-1, 1)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = t1_transformed + upper * torch.exp(s1_transformed)
        z1 = torch.cat([lower_new[:,0].reshape(-1, 1), upper_new, lower_new[:,1].reshape(-1, 1)], dim=1)
       
        lower1, upper1 = z1[:,0].reshape(-1, 1), z1[:,1:3]
        t2_transformed = self.t2(upper1)
        s2_transformed = self.s2(upper1)
        lower_new1 = t2_transformed + lower1 * torch.exp(s2_transformed)
        upper_new1 = upper1
        z2 = torch.cat([lower_new1, upper_new1], dim=1)
        
        lower2, upper2 = z2[:,0::2], z2[:,1].reshape(-1, 1)
        t3_transformed = self.t3(lower2)
        s3_transformed = self.s3(lower2)
        lower_new2 = lower2
        upper_new2 = t3_transformed + upper2 * torch.exp(s3_transformed)
        z3 = torch.cat([lower_new2[:,0].reshape(-1, 1), upper_new2, lower_new2[:,1].reshape(-1, 1)], dim=1)
       
        lower3, upper3 = z3[:,0].reshape(-1, 1), z3[:,1:3]
        t4_transformed = self.t4(upper3)
        s4_transformed = self.s4(upper3)
        lower_new3 = t4_transformed + lower3 * torch.exp(s4_transformed)
        upper_new3 = upper3
        z = torch.cat([lower_new3, upper_new3[:,0].reshape(-1, 1)], dim=1)
        log_det = torch.sum(s1_transformed + s2_transformed + s3_transformed + s4_transformed, dim=1)
        c1 = 1/(2*3.14159265)
        c2 = -torch.sum(z**2, 1)/2
        pz = torch.mul(c1, torch.exp(c2))
        log_px = torch.log(pz) + log_det
        px = torch.exp(log_px)
        return z, log_det, px

    
    def inverse(self, z):
        lower3, upper3 = z[:,0].reshape(-1, 1), z[:,1:3]
        t4_transformed = self.t4(upper3)
        s4_transformed = self.s4(upper3)
        lower_new3 = (lower3 - t4_transformed) * torch.exp(-s4_transformed)
        upper_new3 = upper3 
        x3 = torch.cat([lower_new3[:,0].reshape(-1, 1), upper_new3], dim=1)
        
        lower2, upper2 = x3[:,0::2], x3[:,1].reshape(-1, 1)
        t3_transformed = self.t3(lower2)
        s3_transformed = self.s3(lower2)
        lower_new2 = lower2
        upper_new2 = (upper2 - t3_transformed) * torch.exp(-s3_transformed)        
        x2 =  torch.cat([lower_new2[:,0].reshape(-1, 1), upper_new2, lower_new2[:,1].reshape(-1, 1)], dim=1)
        
        lower1, upper1 = x2[:,0].reshape(-1, 1), x2[:,1:3]
        t2_transformed = self.t2(upper1)
        s2_transformed = self.s2(upper1)
        lower_new1 = (lower1 - t2_transformed) * torch.exp(-s2_transformed)
        upper_new1 = upper1 
        x1 = torch.cat([lower_new1[:,0].reshape(-1, 1), upper_new1], dim=1)
        
        lower, upper = x1[:,0::2], x1[:,1].reshape(-1, 1)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        lower_new = lower
        upper_new = (upper - t1_transformed) * torch.exp(-s1_transformed)        
        x =  torch.cat([lower_new[:,0].reshape(-1, 1), upper_new], dim=1)
        log_det = torch.sum(-s1_transformed - s2_transformed - s3_transformed - s4_transformed, dim=1)
        return x, log_det