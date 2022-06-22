import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs

from utils import Data
from utils.DataFrame import DF 
from model import Autoencoder 

import argparse
import time 

# torch.cuda.empty_cache()
# torch.manual_seed(5)
# torch.cuda.manual_seed(5)
# torch.backends.cudnn.deterministic = True

# Console arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--baseroot', required=True,
                    help='path to training data (images)')
parser.add_argument('-o', '--output_path', required=False,
                    help='path where model files and training information will be placed')
args = parser.parse_args()

# Current Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# LOAD DATA 
transform = {
    'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize((258,540)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]),
    'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize((258,540)),
            transforms.ToTensor(),
            ])}

train_dataset = Data.MyDataset(DF(args.baseroot,'train'), transform['train'], use_cache=False)
test_dataset = Data.MyDataset(DF(args.baseroot,'test'), transform['test'], use_cache=False)

# data loader
batch_size = 8
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, drop_last=True)

# DEFINE MODEL  
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tb = SummaryWriter()

# MAIN LOOP 
worst_loss = 1000
num_epochs = 300
for epoch in range(num_epochs):
    start_time = time.time()
    if epoch == 1:
        trainloader.dataset.set_use_cache(use_cache=True)
        trainloader.num_workers = 4
        testloader.dataset.set_use_cache(use_cache=True)
        testloader.num_workers = 4
    # Train
    model.train()
    train_loss = 0 
    for pair in trainloader:
        imgC, imgN = pair[0].to(device), pair[1].to(device) # [batch,ch,p,h,w]
        optimizer.zero_grad() 
        recon = model(imgN)
        loss = criterion(recon, imgC)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(trainloader.dataset) 
    epoch_time = time.time() - start_time

    # Test 
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for pair in testloader:
            imgC, imgN = pair[0].to(device), pair[1].to(device)
            recon = model(imgN)
            loss = criterion(recon,imgC)
            test_loss += loss.item()
        test_loss /= len(testloader.dataset)

    # tensorboard 
    tb.add_scalars('Loss', {'Train loss':train_loss,'Test loss':test_loss}, epoch)

    # Save model 
    if worst_loss > test_loss:
        worst_loss = test_loss
        state = {'epoch': epoch, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, args.output_path + 'trainModel_11.pth')

    print('Epoch: {}, Train loss: {}, Test loss: {}, time: {:,.2f}'.format(epoch, train_loss, 
            test_loss, epoch_time))
    
tb.close()
