import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs

from utils import data
from utils.dataframe import DF 
from model import Autoencoder 

from decimal import Decimal
import argparse
import time 

torch.cuda.empty_cache()
torch.manual_seed(5)
torch.cuda.manual_seed(5)
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
            transforms.RandomHorizontalFlip(),
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            ]),
    'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            ])}

train_dataset = data.MyDataset(DF(args.baseroot,'train'), transform['train'], None, use_cache=False)
test_dataset = data.MyDataset(DF(args.baseroot,'test'), transform['test'], None, use_cache=False)

# data loader
batch_size = 32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# DEFINE MODEL  
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #weight_decay=1e-3)

tb = SummaryWriter()

# MAIN LOOP 
worst_loss = 1000
num_epochs = 150
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
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
    train_psnr = 0 
    for pair in trainloader:
        imgC, imgN = pair[0].to(device), pair[1].to(device) # [batch,ch,h,w]
        optimizer.zero_grad() 
        recon = model(imgN,device)
        loss = criterion(recon, imgC)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(trainloader.dataset)
    # Peak Signal to Noise Ratio
    train_psnr = 10 * torch.log10(torch.tensor([1]) / train_loss) 
    epoch_time = time.time() - start_time

    # Test 
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_psnr = 0
        for pair in testloader:
            imgC, imgN = pair[0].to(device), pair[1].to(device)
            recon = model(imgN,device)
            loss = criterion(recon,imgC)
            test_loss += loss.item()
        test_loss /= len(testloader.dataset)
        # Peak Signal to Noise Ratio
        test_psnr = 10 * torch.log10(torch.tensor([1]) / test_loss)
    
    # scheduler.step() # update lr 
    
    # tensorboard 
    tb.add_scalars('Loss', {'Train loss':train_loss,'Test loss':test_loss}, epoch)
    tb.add_scalars('PSNR', {'Train psnr':train_psnr.item(),'Test psnr':test_psnr.item()}, epoch)

    # Save model 
    if worst_loss > test_loss:
        worst_loss = test_loss
        state = {'epoch': epoch, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, args.output_path + 'trainmodel_80_20_sin_DP_WD.pth')

    print('Epoch: {}, Train loss: {:.2E}, Test loss: {:.2E}, Train_psnr: {:,.2f} , Test_psnr: {:,.2f}, time: {:,.2f}'.format(epoch, Decimal(train_loss), 
            Decimal(test_loss), train_psnr.item(), test_psnr.item(), epoch_time))
    
tb.close()
