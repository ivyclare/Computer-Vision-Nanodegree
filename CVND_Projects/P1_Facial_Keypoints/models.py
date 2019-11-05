
## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        


        # Convolutional Layers
        #input 1x96x96
        #self.conv1 = nn.Conv2d(1, 32, 4)
        #output size = (W-F)/S +1 = (96-4)/1 +1 = 93
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4)) ##32x93x93, after maxpool layer 32x46x46
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0) ##64x44x44 after maxpool layer 64x22x22
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0) ##128x10x10 after maxpool layer 128x5x5
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0) ##256x5x5 after maxpool layer 256x5x5

        # Pooling Layer for all networks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=136)
        
        # Dropout layer
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        # Initializing Custom weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Convolutional layers have their weights initialized with random numbers drawn from uniform distribution
                m.weight =nn.init.uniform_(m.weight)
                #m.weight = nn.init.uniform(m.weight, a=0, b=1) 
            elif isinstance(m, nn.Linear):
                # FCL layers have weights initialized with Glorot uniform(called Xavier in Pytorch) initialization
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))

        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
    
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))

        x = self.dropout4(self.pool(F.relu(self.conv4(x))))

        ## Flatten images
        x = x.view(x.size(0),-1)

        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x