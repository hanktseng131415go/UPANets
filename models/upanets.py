'''UPANets in PyTorch.

Processing model in cifar10 by Ching-Hsun Tseng and Jia-Nan Feng
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class upa_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.planes = planes
        self.same = same
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU(),
            nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        if l == 1:
            w = 1
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(),
                )
        
        self.att = CPA(in_planes, planes, stride, same=same)
            
    def forward(self, x):

        out = self.cnn(x)
        out = self.att(x, out)

        if self.cat == True:
            out = torch.cat([x, out], 1)
            
        return out

class CPA(nn.Module):
    '''Channel Pixel Attention'''
    
    def __init__(self, in_dim, dim, stride=1, same=False):
        
        super(CPA, self).__init__()
            
        self.dim = dim
        self.stride = stride
        self.same = same
        
        self.cp_ffc = nn.Linear(in_dim, dim)
        self.bn = nn.BatchNorm2d(dim)

        if self.stride == 2 or self.same == True:
            self.cp_ffc_sc = nn.Linear(in_dim, dim)
            self.bn_sc = nn.BatchNorm2d(dim)
            
            if self.stride == 2:
                self.avgpool = nn.AvgPool2d(2)
            
    def forward(self, x, sc_x):        
        
        _, c, w, h = x.shape
        out = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
        out = self.cp_ffc(out)
        out = rearrange(out, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
        out = self.bn(out)  
       
        if out.shape == sc_x.shape:
            out = sc_x + out
            out = F.layer_norm(out, out.size()[1:])
            
        else:
            out = F.layer_norm(out, out.size()[1:])
            x = sc_x
            
        if self.stride == 2 or self.same == True:
            _, c, w, h = x.shape
            x = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
            x = self.cp_ffc_sc(x)
            x = rearrange(x, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
            x = self.bn_sc(x)
            out = out + x 
            
            if self.same == True:
                return out
            
            out = self.avgpool(out)
           
        return out
   
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, img, out=1):
        
        super(SPA, self).__init__()
        
        self.sp_ffc = nn.Sequential(
            nn.Linear(img**2, out**2)
            )   
#        self.sp_ffc = nn.Conv2d(img**2, out**2, kernel_size=1, bias=False)
        
    def forward(self, x):
        
        _, c, w, h = x.shape          
        x = rearrange(x, 'b c w h -> b c (w h)', c=c, w=w, h=h)
        x = self.sp_ffc(x)
        _, c, l = x.shape        
        out = rearrange(x, 'b c (w h) -> b c w h', c=c, w=int(l**0.5), h=int(l**0.5))

        return out
    
class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        
        super(upanets, self).__init__()
        
        self.in_planes = filter_nums
        self.filters = filter_nums
        w = 2
        
        self.root = nn.Sequential(
                nn.Conv2d(3, int(self.in_planes*w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(self.in_planes*w)),
                nn.ReLU(),
                nn.Conv2d(int(self.in_planes*w), self.in_planes*1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
                )        
        self.emb = CPA(3, self.in_planes, same=True)
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], 1)
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], 2)
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], 2)
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], 2)
        
        self.spa0 = SPA(img)
        self.spa1 = SPA(img)
        self.spa2 = SPA(int(img*0.5))
        self.spa3 = SPA(int(img*0.25))
        self.spa4 = SPA(int(img*0.125))

        self.linear = nn.Linear(int(self.filters*31), num_classes)
        self.bn = nn.BatchNorm1d(int(self.filters*31))
     
    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        self.planes = planes
        planes = planes // num_blocks

        for i, stride in enumerate(strides):
            
            if i == 0 and stride == 1:
                layers.append(block(self.planes, self.planes, stride, same=True))
                strides.append(1)
                self.in_planes = self.planes
                
            elif i != 0 and stride == 1:
                layers.append(block(self.in_planes, planes, stride, cat=True))                
                self.in_planes = self.in_planes + planes 
                    
            else:   
                layers.append(block(self.in_planes, self.planes, stride))
                strides.append(1)
                self.in_planes = self.planes
                
        return nn.Sequential(*layers)

    def forward(self, x):
                
        out01 = self.root(x)
        out0 = self.emb(x, out01)
        
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out0_spa = self.spa0(out0)
        out1_spa = self.spa1(out1)
        out2_spa = self.spa2(out2)
        out3_spa = self.spa3(out3)
        out4_spa = self.spa4(out4)
        
        out0_gap = F.avg_pool2d(out0, out0.size()[2:])
        out1_gap = F.avg_pool2d(out1, out1.size()[2:])
        out2_gap = F.avg_pool2d(out2, out2.size()[2:])
        out3_gap = F.avg_pool2d(out3, out3.size()[2:])
        out4_gap = F.avg_pool2d(out4, out4.size()[2:])
      
        out0 = out0_gap + out0_spa
        out1 = out1_gap + out1_spa
        out2 = out2_gap + out2_spa
        out3 = out3_gap + out3_spa
        out4 = out4_gap + out4_spa
        
        out0 = F.layer_norm(out0, out0.size()[1:])
        out1 = F.layer_norm(out1, out1.size()[1:])
        out2 = F.layer_norm(out2, out2.size()[1:])
        out3 = F.layer_norm(out3, out3.size()[1:])
        out4 = F.layer_norm(out4, out4.size()[1:])
        
        out = torch.cat([out4, out3, out2, out1, out0], 1)
        
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.linear(out)

        return out

def UPANets(f, c = 100, block = 1, img = 32):
    
    return upanets(upa_block, [int(4*block), int(4*block), int(4*block), int(4*block)], f, num_classes=c, img=img)

def test():
    
    net = UPANets(16, 10, 1, 64)
    y = net(torch.randn(1, 3, 64, 64))
    print(y.size())

#test()