import numpy as np
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        
        super().__init__()
        
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), #we're preserving input height and width
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), #we're preserving input height and width
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)
    


class UNET(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512,1024]):
        
        super().__init__()
        
        self.downsamples=nn.ModuleList()
        
        self.upsamples= nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #now building the encoder
        
        for feature in features:
            self.downsamples.append(DoubleConv(in_channels, feature))
            in_channels=feature

        #the decoder
        
        for feature in reversed(features):
            self.upsamples.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upsamples.append(DoubleConv(feature*2, feature))
            
        #bottleneck layer
        self.bottleneck= DoubleConv(features[-1], features[-1]*2)

        #final 1x1 conv to change out_channels
        self.final= nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
    def forward(self,x):
        skip_connections=[]

        for downsampler in self.downsamples:

            x = downsampler(x) #this is the output of each downsampling layer
            skip_connections.append(x)
            x= self.pool(x)

        x= self.bottleneck(x) 

        skip_connections= skip_connections[::-1]

        for i in range(0, len(self.upsamples), 2):

            x= self.upsamples[i](x) # this is the conv transpose layer
            skipped= skip_connections[i//2]

            concat_skipped= torch.cat((skipped,x),dim=1)

            x=self.upsamples[i+1](concat_skipped) #this is the double conv layer

        return self.final(x)