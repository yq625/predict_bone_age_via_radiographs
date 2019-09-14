#!/bin/python

#module load pytorch/intel/20170724
#module load scikit-image/intel/0.13.1
#module load pillow/intel/4.0.0
#module load torchvision/0.1.8

#python

import torch
from torch import nn, autograd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
from PIL import Image
import time
import os
from skimage import io, color

import torchvision
from torchvision import transforms, models

        # image = io.imread(img_name,as_gray=True) 

class Boneagedataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.resize((512,512))
        image = np.asarray(image)
        image = (image - image.mean()) / image.std()
        image = torch.cuda.FloatTensor(image)
        image = image.repeat(3,1,1)
        
        image_bone_age = self.data_frame.iloc[idx, 1]
        image_bone_age = torch.cuda.FloatTensor([image_bone_age])
        
        image_sex = self.data_frame.iloc[idx, 2]
        image_sex = torch.cuda.FloatTensor([image_sex])
        if self.transform:
            image = self.transform(image)
        sample = {'x': image, 'y': image_bone_age, 'z': image_sex}
        return sample
        

dataset = {'train': Boneagedataset('~/boneagechallenge/random_train_boneage.csv', '/beegfs/ga4493/projects/team_G/boneage-training-dataset_final'),
           'validate':Boneagedataset('~/boneagechallenge/random_validation_boneage.csv', '/beegfs/ga4493/projects/team_G/boneage-training-dataset_final'),
          }

bs = 4

dataloader = {x: DataLoader(dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate']}      
 
 
			
			
def train_model(model, dataloader, optimizer, loss_fn, num_epochs = 10, verbose = False):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_loss = 100
    phases = ['train','validate']
    since = time.time()
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0
            if p == 'train':
                model.train(True)
            else:
                model.train(False)                
            for data in dataloader[p]:
                optimizer.zero_grad()
                image = autograd.Variable(data['x'])
                label = autograd.Variable(data['y'])
                sex = autograd.Variable(data['z'])
                output = model(image, sex)
                loss = loss_fn(output, label)
                num_imgs = image.size()[0]
                running_loss += loss.data.cpu().numpy()[0]*num_imgs
                running_total += num_imgs
                if p== 'train':
                    loss.backward()
                    optimizer.step()
            epoch_loss = float(running_loss)/float(running_total)
            #if verbose or (i%10 == 0):
            print('Phase:{}, epoch loss: {:.4f} '.format(p, epoch_loss))
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
        if (i%10 == 0):
            torch.save(best_model_wts, 'resnet34_all_20ep_sex.wts')
            time_elapsed = time.time() - since
            print('10 epoch costs {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))    
    model.load_state_dict(best_model_wts)    
    return model, loss_dict
	

	
resnet = models.resnet34(pretrained=True)

class NET(nn.Module):
    def __init__(self , model):
        super(NET,self).__init__()
        self.resnet_features = nn.Sequential(*list(model.children())[:-1])
        #takes in 3*512*512, outputs 512*16*16
        self.avg_pool = nn.AvgPool2d(kernel_size=16)
        self.fc = nn.Sequential(nn.Linear(513, 50), nn.Linear(50, 1))
    def forward(self, x, sex):
        x = self.resnet_features(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = torch.cat((x,sex), dim=1)
        x = self.fc(x)  
        return x


net = NET(resnet)
net.cuda()

	
loss_fn = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
best_model, loss_dict = train_model(net, dataloader, optimizer, loss_fn, num_epochs = 50)

	
torch.save(best_model.state_dict(), 'resnet34_all_50ep_sex.pkl')

        
with open('loss_train_resnet34_all_50ep_sex.txt', 'w') as f:
        f.write(str(loss_dict['train']))
with open('loss_validate_resnet34_all_50ep_sex.txt', 'w') as f:
        f.write(str(loss_dict['validate']))

