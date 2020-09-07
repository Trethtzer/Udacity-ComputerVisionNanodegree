## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## output size = (W-F)/S +1 = ((224-5)/1 +1)/2 = 110
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv1drop = nn.Dropout(p=0.1)
        
        ## output size = (W-F)/S +1 = ((110-3)/1 +1)/2 = 54
        self.conv2 = nn.Conv2d(32,128,3)
        self.conv2drop = nn.Dropout(p=0.2)
        
        ## output size = (W-F)/S +1 = ((54-3)/1 +1)/2 = 26
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv3drop = nn.Dropout(p=0.3)      
        
        ## output size = (W-F)/S +1 = ((26-3)/1 +1)/2 = 12
        self.conv4 = nn.Conv2d(256,512,3)
        self.conv4drop = nn.Dropout(p=0.4)
        
        ## output size = (W-F)/S +1 = ((12-2)/1 +1)/2 = 5
        self.conv5 = nn.Conv2d(512,1024,2)
        self.conv5drop = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(5*5*1024,1500)
        self.fc1drop = nn.Dropout(p=0.5)
        
        #self.fc2 = nn.Linear(1000,1000)
        #self.fc2drop = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(1500,136)
        
        self.pool = nn.MaxPool2d(2,2)
        
       
        
    def forward(self, x):
        
        x = self.conv1drop(self.pool(F.elu(self.conv1(x))))
        x = self.conv2drop(self.pool(F.elu(self.conv2(x))))
        x = self.conv3drop(self.pool(F.elu(self.conv3(x))))
        x = self.conv4drop(self.pool(F.elu(self.conv4(x))))
        x = self.conv5drop(self.pool(F.elu(self.conv5(x))))
        
        # Flatten 
        x = x.view(x.size(0), -1)
        
        x = self.fc1drop(F.elu(self.fc1(x)))        
        #x = self.fc2drop(F.elu(self.fc2(x))) 
        x = self.fc3(x)
        

        return x
