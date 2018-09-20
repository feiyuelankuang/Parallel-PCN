'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv2d(nn.Module):
    def __init__(self, inchan, outchan, sample=False):
        super().__init__()
        self.kernel_size = 3
        self.weights = nn.init.xavier_normal(torch.Tensor(outchan,inchan,self.kernel_size,self.kernel_size))
        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.sample = sample
    
        if self.sample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x, feedforward=True):
        if feedforward:
            x = F.conv2d(x, self.weights, stride=1, padding=1)              
            if self.sample:
                x = self.Downsample(x)
        else:
            if self.sample:
                x = self.Upsample(x)
            x = F.conv_transpose2d(x, self.weights, stride=1, padding=1)
        return x


class PredictiveTied(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.1) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.1) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](xr[t-2]-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](xr[-2]-P[-2])*b0)
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                state_list.append(xr[self.layer_sel])
                error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, error_list
        else:
            return out#, state_list


class PredictiveTied2(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.1) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.1) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])

                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])

                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])

                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list

class PredictiveTied_penalty(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.01) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.01) for i in range(self.nlays)])
        self.c0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.01) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            c0 = F.relu(self.c0[0]).expand_as(xr[0])
            xr[0] = F.relu((1-c0)*xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                c0 = F.relu(self.c0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0] -c0*xr[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i] -c0*xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                c0 = F.relu(self.c0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu((1-c0)*xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]-c0*xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                c0 = F.relu(self.c0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu((1-c0)*xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]-c0*xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                c0 = F.relu(self.c0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu((1-c0)*xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                c0 = F.relu(self.c0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0]-c0*xr[0])              
               
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]-c0*xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                c0 = F.relu(self.c0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu((1-c0)*xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                
               
                
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]-c0*xr[i])
                
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                c0 = F.relu(self.c0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu((1-c0)*xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    c0 = F.relu(self.c0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]-c0*xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                c0 = F.relu(self.c0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu((1-c0)*xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list

class PredictiveTied3(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,1,1)+0.1) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,1,1,1)+0.1) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0
            
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = (1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0]
                

                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = xr[t-1]+self.conv[t-1](old-P[t-2])*b0
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = xr[t-1]+self.conv[t-1](old-P[t-2])*b0
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+t+1]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = (1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0]              
                
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = xr[t-1]+self.conv[t-1](old-P[t-2])*b0
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = (1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0]
                              
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = (1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, error_list
        else:
            return out#, state_list

class PredictiveTiedLayerNorm(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False, affine = True, a = 0.5, b = 0.5):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        self.affine = affine
        self.a = a
        self.b = b
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+self.a) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+self.b) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls == 1:
            for i in range(1,self.nlays):
                xr.append(F.relu(self.conv[i](xr[-1])))
            out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
            out = out.view(out.size(0), -1)
            #torch.nn.Dropout()
            out = self.linear(out)
            return out#, state_list  
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
#            print(xr[0].size())
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
#            print(xr[0].size())
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
 #               print(xr[0].size())
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] =m(xr[0])
 #               print(xr[0].size())         
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1+t]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
#                    print(xr[i].size())
 #                   print(tmp.size())
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list

class PredictiveTiedInstanceNorm(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False, affine = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        self.affine = affine
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.1) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.1) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        m = nn.InstanceNorm2d(xr[0].size()[1], affine=True).cuda()
        xr[0] = m(xr[0])    
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            m = nn.InstanceNorm2d(xr[0].size()[1], affine=True).cuda()
            xr[0] = m(xr[0])
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
               
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
                m = nn.InstanceNorm2d(xr[0].size()[1], affine=True).cuda()
                xr[0] = m(xr[0])                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.InstanceNorm2d(xr[i].size()[1], affine=True).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.InstanceNorm2d(xr[t-1].size()[1], affine=True).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.InstanceNorm2d(xr[i].size()[1], affine=True).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.InstanceNorm2d(xr[t-1].size()[1], affine=True).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.InstanceNorm2d(xr[i].size()[1], affine=True).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.InstanceNorm2d(xr[-1].size()[1], affine=True).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                               
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                
                               
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list


class PredictiveTiedKL(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
#        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.1) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.1) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]-b0*self.conv[0](torch.log(self.conv[0](xr[0],feedforward=False)/x+1.0)))
#            print(xr[0].size())
#            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
#            xr[0] = m(xr[0])
#            print(xr[0].size())
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
 #               a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu(xr[0]-b0*self.conv[0](torch.log(self.conv[0](xr[0],feedforward=False)/x+1.0)))
 #               print(xr[0].size())
               
 #               print(xr[0].size())         
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
   #                 a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu(xr[i]-b0*self.conv[i](torch.log(self.conv[i](xr[i],feedforward=False)/tmp+1.0))+ b0*self.conv[i+1](xr[i+1],feedforward=False)/xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]-b0*self.conv[t-1](torch.log(self.conv[t-1](xr[t-1],feedforward=False)/old+1.0)))
  #              xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
    #                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu(xr[i]-b0*self.conv[i](torch.log(self.conv[i](xr[i],feedforward=False)/tmp+1.0))+ b0*self.conv[i+1](xr[i+1],feedforward=False)/xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]-b0*self.conv[t-1](torch.log(self.conv[t-1](xr[t-1],feedforward=False)/old+1.0)))
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
    #                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu(xr[i]-b0*self.conv[i](torch.log(self.conv[i](xr[i],feedforward=False)/tmp+1.0))+ b0*self.conv[i+1](xr[i+1],feedforward=False)/xr[i])                 
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]-b0*self.conv[-1](torch.log(self.conv[-1](xr[-1],feedforward=False)/old+1.0)))
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
        #        a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                           
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                #    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                           
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                 #   a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    #a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list

class PredictiveTied_LN_Multi(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False, affine = True, a = 0.1, b = 0.1):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        self.affine = affine
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+a) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+b) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
        xr[-1] = m(xr[-1])
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls == 1:
            for i in range(1,self.nlays):
                xr.append(F.relu(self.conv[i](xr[-1])))
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
            out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
            out = out.view(out.size(0), -1)
            #torch.nn.Dropout()
            out = self.linear(out)
            return out#, state_list  
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
            xr[-1] = m(xr[-1])
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
#            print(xr[0].size())
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
            out=[]
#            print(xr[0].size())
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
 #               print(xr[0].size())
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] =m(xr[0])
 #               print(xr[0].size())         
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            tmp = xr[-1]/self.cls;
            tmp = F.avg_pool2d(tmp, tmp.size(-1))
            tmp = tmp.view(tmp.size(0), -1)
            #torch.nn.Dropout()
            out.append(self.linear(tmp))
                
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1+t]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
#                    print(xr[i].size())
 #                   print(tmp.size())
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                tmp = xr[-1]/self.cls;
                tmp = F.avg_pool2d(tmp, tmp.size(-1))
                tmp = tmp.view(tmp.size(0), -1)
                #torch.nn.Dropout()
                out.append(self.linear(tmp))
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                
        if self.record:
            return out, error_list
        else:
            return out#, state_list


class PredictiveTied_LN(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False, affine = True, a = 0.1, b = 0.1):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        self.affine = affine
        self.a = a
        self.b = b
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+self.a) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+self.b) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
        xr[0] =m(xr[0])
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls == 1:
            for i in range(1,self.nlays):
                xr.append(F.relu(self.conv[i](xr[-1])))
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] =m(xr[-1])
            out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
            out = out.view(out.size(0), -1)
            #torch.nn.Dropout()
            out = self.linear(out)
            return out#, state_list  
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
            xr[-1] =m(xr[-1])
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
#            print(xr[0].size())
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
#            print(xr[0].size())
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] =m(xr[-1])
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])
 #               print(xr[0].size())
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] =m(xr[0])
 #               print(xr[0].size())         
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1+t]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
#                    print(xr[i].size())
 #                   print(tmp.size())
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
            xr[-1] = m(xr[-1])
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list

class PredictiveTied_GN(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =0, record = False, affine = True, a = 0.1, b = 0.1, num_GN = 1):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.record = record
        self.affine = affine
        self.a = a
        self.b = b
        # Feedforward layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+self.a) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+self.b) for i in range(self.nlays)])
        self.GN = nn.ModuleList([nn.GroupNorm(num_GN,ocs[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.conv[0](x))]
        xr[0] = self.GN[0](xr[0])
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls == 1:
            for i in range(1,self.nlays):
                xr.append(F.relu(self.conv[i](xr[-1])))              
                xr[-1] =self.GN[i](xr[-1])
            out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
            out = out.view(out.size(0), -1)
            #torch.nn.Dropout()
            out = self.linear(out)
            return out#, state_list  
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            xr[-1] = self.GN[1](xr[-1])
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = self.GN[0](F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0))
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
              
                xr.append(F.relu(self.conv[t](xr[-1])))
                xr[-1] = self.GN[t](xr[-1])
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = self.GN[0](F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0]))   
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = self.GN[i](F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]))
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = self.GN[t-1](F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0))
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                      
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(self.GN[t](F.relu(self.conv[t](xr[-1]))))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = self.GN[i](F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]))
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = self.GN[t-1](F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0))
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[self.nlays-self.cls+1+t]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
#                    print(xr[i].size())
 #                   print(tmp.size())
                    xr[i] = self.GN[i](F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i]))
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = self.GN[-1](F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0))
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.conv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0)
            m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
            xr[0] = m(xr[0])
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.conv[self.layer_sel+1](xr[self.layer_sel+1],feedforward=False))
            for t in range(2,self.nlays):
                xr.append(F.relu(self.conv[t](xr[-1])))  
                for i in range(t):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)                       
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])              
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.conv[t-1](old-P[t-2])*b0)
                m = nn.LayerNorm(xr[t-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[t-1] = m(xr[t-1])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
 
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.conv[i+1](xr[i+1],feedforward=False)
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.conv[0](x-self.conv[0](xr[0],feedforward=False))*b0) + a0*P[0])                
                m = nn.LayerNorm(xr[0].size()[1:],elementwise_affine=self.affine).cuda()
                xr[0] = m(xr[0])               
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                state_list.append(xr[self.layer_sel])
                if self.layer_sel < self.nlays - 1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
            #stage 3
            for t in range(self.nlays-1):
                for i in range(t,self.nlays-1):
                    P[i] = self.conv[i+1](xr[i+1],feedforward=False)
                old = xr[t]
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.conv[i](tmp-P[i-1])*b0) + a0*P[i])
                    m = nn.LayerNorm(xr[i].size()[1:],elementwise_affine=self.affine).cuda()
                    xr[i] = m(xr[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.conv[self.nlays-1](old-P[-2])*b0)
                m = nn.LayerNorm(xr[-1].size()[1:],elementwise_affine=self.affine).cuda()
                xr[-1] = m(xr[-1])
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1 and self.layer_sel < self.nlays-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        if self.record:
            return out, state_list
        else:
            return out#, state_list


