'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Feedforeward module
class FFconv2d(nn.Module):
    def __init__(self, inchan, outchan, downsample=False, norm = ''):
        super().__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
   #     self.BatchNorm = nn.BatchNorm2d(outchan)
   #     self.LRN = nn.LocalResponseNorm(2)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
    #    x = self.BatchNorm(x)
    #    x = self.LRN(x)
#        x = nn.LayerNorm(x.size()[1:],elementwise_affine=False)
        if norm == 'InstanceNorm':
            x = nn.InstanceNorm2d(x.size()[1])
        if self.downsample:
            x = self.Downsample(x)

        return x


# Feedback module
class FBconv2d(nn.Module):
    def __init__(self, inchan, outchan, upsample=False, norm = ''):
        super().__init__()
        self.convtranspose2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
      #  self.BatchNorm = nn.BatchNorm2d(outchan)
#        self.LRN = nn.LocalResponseNorm(2)
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            
    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtranspose2d(x)
     #   self.BatchNorm = nn.BatchNorm2d(outchan)
 #       x = nn.LayerNorm(x.size()[1:],elementwise_affine=False)
        if norm == 'InstanceNorm':
            x = nn.InstanceNorm2d(x.size()[1])
 #       x = self.LRN(x)
        return x



# PredNet
class PredNet(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics)

        # Feedforward layer
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        for i in range(1,self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[i-1])))

        # Dynamic process 
        for t in range(self.cls):
            # Feedback prediction
            xp = []
            for i in range(self.nlays-1,0,-1):
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a0 + xr[i-1]*(1-a0))
            # Feedforward prediction error
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            for i in range(1, self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-xp[i-1])*b0 + xr[i])

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PredNetNew(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]

        for i in range(1,self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[i-1])))
            a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
            xr[i-1] = F.relu(self.FBconv[i](xr[i])*a0 + xr[i-1]*(1-a0))

        for t in range(self.cls):
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            for i in range(1,self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 + xr[i])
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(self.FBconv[i](xr[i])*a0 + xr[i-1]*(1-a0))

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PredNetNewRead(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x, layer_sel):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        state_list = []

        for i in range(1,self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[i-1])))
            a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
            xr[i-1] = F.relu(self.FBconv[i](xr[i])*a0 + xr[i-1]*(1-a0))
        state_list.append(xr[layer_sel])

        for t in range(self.cls):
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            for i in range(1,self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 + xr[i])
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(self.FBconv[i](xr[i])*a0 + xr[i-1]*(1-a0))
            state_list.append(xr[layer_sel])
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, state_list


class Predictive(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        if 2*self.cls <= self.nlays:
            xr.append(F.relu(self.FFconv[1](xr[0])))
            #stage 1
            for t in range(1,self.cls):
                xr.append(F.relu(self.FFconv[2*t](xr[-1])))
                for i in range(2,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*self.FBconv[1](xr[1]))
                xr.append(F.relu(self.FFconv[2*t+1](xr[-1])))
                for i in range(1,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
            #stage 2 
            for t in range(self.nlays//2-self.cls):
                xr.append(F.relu(self.FFconv[2*t+2*self.cls](xr[-1])))
                for i in range(2*t+2,2*self.cls+2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))

                xr.append(F.relu(self.FFconv[2*t+1+2*self.cls](xr[-1])))
                for i in range(2*t+3,2*self.cls+2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))

             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-2*(self.cls-1-t),self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))

                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(self.nlays-2*(self.cls-1-t)+1,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))

        elif 2*self.cls > self.nlays:
#stage  1
            xr.append(F.relu(self.FFconv[1](xr[0])))
            for t in range(1,self.nlays//2):
                xr.append(F.relu(self.FFconv[2*t](xr[-1])))
                for i in range(2,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*self.FBconv[1](xr[1]))
                xr.append(F.relu(self.FFconv[2*t+1](xr[-1])))
                for i in range(1,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
        #stage 2
            for t in range(self.cls-self.nlays//2):
                for i in range(2,self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*self.FBconv[1](xr[1]))

                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(1,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))
            #stage 3
            for t in range(self.nlays//2-1):
                for i in range(2*t+2,self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i])
                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(2*t+3,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0) + a0*self.FBconv[i+1](xr[i+1]))

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PredNetLocal(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        if 2*self.cls <= self.nlays:
            xr.append(F.relu(self.FFconv[1](xr[0])))
            #stage 1
            for t in range(1,self.cls):
                xr.append(F.relu(self.FFconv[2*t](xr[-1])))
                for i in range(2,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 - a0*(self.FBconv[1](xr[1])-xr[0]))
                xr.append(F.relu(self.FFconv[2*t+1](xr[-1])))
                for i in range(1,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            #stage 2 
            for t in range(self.nlays//2-self.cls):
                xr.append(F.relu(self.FFconv[2*t+2*self.cls](xr[-1])))     
                for i in range(2*t+2,2*self.cls+2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
                xr.append(F.relu(self.FFconv[2*t+1+2*self.cls](xr[-1])))
                for i in range(2*t+3,2*self.cls+2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))

            #stage 3
            for t in range(self.cls-1):                
                for i in range(self.nlays-2*(self.cls-1-t),self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(self.nlays-2*(self.cls-1-t)+1,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
        elif 2*self.cls > self.nlays:
#stage  1
            xr.append(F.relu(self.FFconv[1](xr[0])))
            for t in range(1,self.nlays//2):
                xr.append(F.relu(self.FFconv[2*t](xr[-1])))
                for i in range(2,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 - a0*(self.FBconv[1](xr[1])-xr[0]))
                xr.append(F.relu(self.FFconv[2*t+1](xr[-1])))
                for i in range(1,2*t,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
        #stage 2
            for t in range(self.cls-self.nlays//2):
                for i in range(2,self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 - a0*(self.FBconv[1](xr[1])-xr[0]))
                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(1,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))

        #stage 3
            for t in range(self.nlays//2-1):
                for i in range(2*t+2,self.nlays,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i])
                b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
                xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
                for i in range(2*t+3,self.nlays-1,2):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    
class PredNetLocalRead(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x,layer_sel = 1):
        # Feedforward
        state_list = []
        xr = [F.relu(self.FFconv[0](x))]
        xr.append(F.relu(self.FFconv[1](xr[0])))
        if layer_sel < 2:
            state_list.append(xr[layer_sel])
        for t in range(1,self.nlays//2):
            xr.append(F.relu(self.FFconv[2*t](xr[-1])))
            for i in range(2,2*t,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            a0 = F.relu(self.a0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 - a0*(self.FBconv[1](xr[1])-xr[0]))
            xr.append(F.relu(self.FFconv[2*t+1](xr[-1])))
            for i in range(1,2*t,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            if layer_sel < 2*t+2:
                state_list.append(xr[layer_sel])
        #stage 2
        for t in range(self.cls-self.nlays//2):
            for i in range(2,self.nlays,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            a0 = F.relu(self.a0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 - a0*(self.FBconv[1](xr[1])-xr[0]))
            b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
            xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
            for i in range(1,self.nlays-1,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            state_list.append(xr[layer_sel])
        
        #stage 3
        for t in range(self.nlays//2-1):
            for i in range(2*t+2,self.nlays,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i])
            b0 = F.relu(self.b0[self.nlays-1]).expand_as(xr[-1])
            xr[-1] = F.relu(self.FFconv[self.nlays-1](xr[-2]-self.FBconv[self.nlays-1](xr[-1]))*b0 + xr[-1])
            for i in range(2*t+3,self.nlays-1,2):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i]))*b0 - a0*(self.FBconv[i+1](xr[i+1])-xr[i]))
            if layer_sel > 2*t+1:
                state_list.append(xr[layer_sel])
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,state_list

class PredictiveFull(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        P = [None for i in range(self.nlays)]

        for i in range(1,self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[-1])))
                  
        for t in range(self.cls):
            for i in range(self.nlays-1):
                P[i] =  self.FBconv[i+1](xr[i+1])
              
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            a0 = F.relu(self.a0[0]).expand_as(xr[0])
            xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
            b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
            xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
    
            for i in range(1,self.nlays-1):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                a0 = F.relu(self.a0[i]).expand_as(xr[i])
                xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
               
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PredictiveNew(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =2):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.01) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.01) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        state_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.FFconv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
              
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])

                  
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                for i in range(t-self.cls, t-1):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.FBconv[i+1](xr[i+1])
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])

             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+1+t,self.nlays-1):
                    P[i] = self.FBconv[i+1](xr[i+1])
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.FFconv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            for t in range(2,self.nlays):  
                for i in range(t-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
              
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                state_list.append(xr[self.layer_sel])

            #stage 3
            for t in range(self.nlays-1):
                for i in range(t+1,self.nlays-1):
                    P[i] = self.FBconv[i+1](xr[i+1])
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        return out#, state_list

class PredictiveNew2(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.01) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.01) for i in range(self.nlays)])
        #self.LRN = nn.ParameterList([nn.LocalResponseNorm(2) for i in range(self.nlays*2)])

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = 0
#        for i in range(self.cls):
#            torch.nn.init.xavier_uniform(self.FFconv[i].conv2d.weight)
            #torch.nn.init.xavier_uniform(self.FBconv[i].convtranspose2d.weight)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        state_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.FFconv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            #if self.layer_sel < 2:
            #    state_list.append(xr[self.layer_sel])
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
              
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    xr[i] = F.relu(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])

                  
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                for i in range(t-self.cls, t-1):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.FBconv[i+1](xr[i+1])
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    xr[i] = F.relu(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0)
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
  
             
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+1+t,self.nlays-1):
                    P[i] = self.FBconv[i+1](xr[i+1])
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    xr[i] = F.relu(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0)
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])

        
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        return out#, state_list


class PredictiveComb(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel = 2):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)
        self.layer_sel = layer_sel

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
            self.comb = nn.ModuleList([nn.Conv3d(ocs[i], ocs[i], kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1), bias=True) for i in range(self.nlays-1)])
            self.comblast = nn.Conv3d(ocs[-1], ocs[-1], kernel_size=(2,3,3), stride=(1,1,1), padding=(0,1,1), bias=True) 
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.Parameter(torch.zeros(1,ocs[self.nlays-1],1,1)+1.0)
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
 #       print(xr[0])
        P = xr[-1]
        P[P!=0] = 0
        P = [P]
#        print(P[0])
 #       print(P)
        for i in range(1,self.nlays):
            xr.append(F.relu(self.FFconv[i](xr[-1])))
            P.append(F.relu(self.FFconv[i](P[-1])))
        
        state_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])


        for j in range(self.cls):
          #  print('time')
          #  print(j)
            for i in range(self.nlays-1):
                #print(xr[i+1].size())
                P[i] =  self.FBconv[i+1](xr[i+1])
                #print(i)
            xr[0] = F.relu(self.comb[0](torch.cat((xr[0].unsqueeze(2),P[0].unsqueeze(2),(self.FFconv[0](x-self.FBconv[0](xr[0]))).unsqueeze(2)),2)).squeeze())
            for i in range(1,self.nlays-1):
        #        print(i)
        #        print(xr[i].size())
        #        print(xr[i-1].size())
        #        print(self.FBconv[i](xr[i]).size())
                xr[i] = F.relu(self.comb[i](torch.cat((xr[i].unsqueeze(2),P[i].unsqueeze(2),self.FFconv[i](xr[i-1]-self.FBconv[i](xr[i])).unsqueeze(2)),2)).squeeze())
            xr[self.nlays-1] = F.relu(self.comblast(torch.cat((xr[-1].unsqueeze(2),self.FFconv[self.nlays-1](xr[-1]-self.FBconv[self.nlays-1](xr[-1])).unsqueeze(2)),2)).squeeze())   
               
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PredictiveSELU(nn.Module):
    def __init__(self, inum_classes=10, cls=3,layer_sel=2):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.selu = nn.SELU()
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel

    def forward(self, x):
        # Feedforward
        xr = [self.selu(self.FFconv[0](x))]
        state_list=[]
        error_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
      
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(self.selu(self.FFconv[1](xr[-1])))
            b0 = self.selu(self.b0[0]).expand_as(xr[0])
            xr[0] = self.selu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.selu(self.FBconv[self.layer_sel+1](xr[self.layer_sel+1])))
            #after t>1
            for t in range(2,self.cls):
                xr.append(self.selu(self.FFconv[t](xr[-1])))  
                for i in range(t):
                    P[i] = self.selu(self.FBconv[i+1](xr[i+1]))
             
                b0 = self.selu(self.b0[0]).expand_as(xr[0])
                a0 = self.selu(self.a0[0]).expand_as(xr[0])
                xr[0] = self.selu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = self.selu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = self.selu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                  
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(self.selu(self.FFconv[t](xr[-1])))
                for i in range(t-self.cls, t):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.selu(self.FBconv[i+1](xr[i+1]))
                for i in range(t-self.cls+1,t-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = self.selu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = self.selu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t and self.layer_sel > t-self.cls-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
             
             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+t,self.nlays-1):
                    P[i] = self.selu(self.FBconv[i+1](xr[i+1]))
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = self.selu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = self.selu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > self.nlays -self.cls:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
        else:
            #stage 1
            xr.append(self.selu(self.FFconv[1](xr[-1])))
            b0 = self.selu(self.b0[0]).expand_as(xr[0])
            xr[0] = self.selu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            if self.layer_sel < 1:
                error_list.append(xr[self.layer_sel]-self.selu(self.FBconv[self.layer_sel+1](xr[self.layer_sel+1])))
            
            
            for t in range(2,self.nlays):  
                xr.append(self.selu(self.FFconv[t](xr[-1])))

                for i in range(t):
                    P[i] =  self.selu(self.FBconv[i+1](xr[i+1]))
              
                b0 = self.selu(self.b0[0]).expand_as(xr[0])
                a0 = self.selu(self.a0[0]).expand_as(xr[0])
                xr[0] = self.selu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = self.selu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = self.selu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel < t:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])

            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.selu(self.FBconv[i+1](xr[i+1]))
                b0 = self.selu(self.b0[0]).expand_as(xr[0])
                a0 = self.selu(self.a0[0]).expand_as(xr[0])
                xr[0] = self.selu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = self.selu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = self.selu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                
                for i in range(1,self.nlays-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                state_list.append(xr[self.layer_sel])
                error_list.append(xr[self.layer_sel]-P[self.layer_sel])

            #stage 3
            for t in range(self.nlays-1):
                for i in range(t+1,self.nlays-1):
                    P[i] = self.selu(self.FBconv[i+1](xr[i+1]))
                for i in range(t+1,self.nlays-1):
                    b0 = self.selu(self.b0[i]).expand_as(xr[i])
                    a0 = self.selu(self.a0[i]).expand_as(xr[i])
                    xr[i] = self.selu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = self.selu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = self.selu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                if self.layer_sel > t-1:
                    error_list.append(xr[self.layer_sel]-P[self.layer_sel])
                
        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        return out, state_list

class PredictiveNew3(nn.Module):
    def __init__(self, num_classes=10, cls=3,layer_sel =2):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # circles means number of additional inputs
        self.nlays = len(ics)

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])
        # Update rate
        #self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        #self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.01) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+0.01) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)
        self.layer_sel = layer_sel
    def forward(self, x):
        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        state_list=[]
        if self.layer_sel < 1:
            state_list.append(xr[self.layer_sel])
        P = [None for i in range(self.nlays)]
        if self.cls <= self.nlays:
            #stage 1
            #t = 1
            xr.append(F.relu(self.FFconv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            #after t>1
            for t in range(2,self.cls):  
                for i in range(t-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
              
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                old = xr[0]
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](tmp-P[i-1])*b0) + a0*P[i])
                    

                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])

                  
                   
            #stage 2 
            for t in range(self.cls,self.nlays):
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                for i in range(t-self.cls, t-1):
#                    print('len:',len(P))
 #                   print(i)
                    P[i] =  self.FBconv[i+1](xr[i+1])
                old = xr[t-self.cls]
                for i in range(t-self.cls+1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](tmp-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](old-P[t-2])*b0)
                if self.layer_sel < t+1 and self.layer_sel > t-self.cls:
                    state_list.append(xr[self.layer_sel])

             #stage 3
            for t in range(self.cls-1):
                for i in range(self.nlays-self.cls+1+t,self.nlays-1):
                    P[i] = self.FBconv[i+1](xr[i+1])
                old = xr[self.nlays-self.cls+1+t]
                for i in range(self.nlays-self.cls+2+t,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    tmp = old
                    old = xr[i]
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](tmp-P[-2])*b0)
                if self.layer_sel > self.nlays-self.cls+1:
                    state_list.append(xr[self.layer_sel])

        else:
            #stage 1
            xr.append(F.relu(self.FFconv[1](xr[-1])))
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0)
            if self.layer_sel < 2:
                state_list.append(xr[self.layer_sel])
            for t in range(2,self.nlays):  
                for i in range(t-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
              
                xr.append(F.relu(self.FFconv[t](xr[-1])))
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[t-1]).expand_as(xr[t-1])
                xr[t-1] = F.relu(xr[t-1]+self.FFconv[t-1](xr[t-2]-P[t-2])*b0)
                for i in range(1,t-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                if self.layer_sel < t+1:
                    state_list.append(xr[self.layer_sel])
            #stage 2
            for t in range(self.cls-self.nlays):
                for i in range(self.nlays-1):
                    P[i] =  self.FBconv[i+1](xr[i+1])
                b0 = F.relu(self.b0[0]).expand_as(xr[0])
                a0 = F.relu(self.a0[0]).expand_as(xr[0])
                xr[0] = F.relu((1-a0)*(xr[0]+self.FFconv[0](x-self.FBconv[0](xr[0]))*b0) + a0*P[0])
                
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                
                for i in range(1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                state_list.append(xr[self.layer_sel])

            #stage 3
            for t in range(self.nlays-1):
                for i in range(t+1,self.nlays-1):
                    P[i] = self.FBconv[i+1](xr[i+1])
                for i in range(t+1,self.nlays-1):
                    b0 = F.relu(self.b0[i]).expand_as(xr[i])
                    a0 = F.relu(self.a0[i]).expand_as(xr[i])
                    xr[i] = F.relu((1-a0)*(xr[i]+self.FFconv[i](xr[i-1]-P[i-1])*b0) + a0*P[i])
                b0 = F.relu(self.b0[-1]).expand_as(xr[-1])
                xr[-1] = F.relu(xr[-1]+self.FFconv[self.nlays-1](xr[-2]-P[-2])*b0)
                if self.layer_sel > t:
                    state_list.append(xr[self.layer_sel])
                

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        #torch.nn.Dropout()
        out = self.linear(out)
        return out#, state_list
