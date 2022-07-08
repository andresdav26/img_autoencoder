import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import Autoencoder 
from utils import data
from utils.dataframe import DF 

from itertools import product
from decimal import Decimal
import argparse
import time
import cv2

# Console arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True,
                    help='path to model')
parser.add_argument('-r', '--baseroot', required=True,
                    help='path to val data (images)')
parser.add_argument('-o', '--output_path', required=False,
                    help='path where model files and training information will be placed')
args = parser.parse_args()

# Load data 
batch_size = 1
transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            ])
            
val_dataset = data.MyDataset(DF(args.baseroot,'val'), transform, use_cache=False)
valloader = DataLoader(val_dataset, batch_size=batch_size)

# Current Device 
device = 'cpu'
print('Using device:', device)

# Define model
model = Autoencoder().to(device)

# load model
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.MSELoss()

with torch.no_grad():
        model.eval()
        for k, pair in enumerate(valloader):
            start_time = time.time()
            imgC, imgN = pair[0].to(device), pair[1].to(device)

            recon = model(imgN,device)
            #reconT[:,:,i:i+d,j:j+d] = recon
            loss = criterion(recon,imgC)
            recon_time = time.time() - start_time
            
            # Peak Signal to Noise Ratio
            psnr = 10 * torch.log10(torch.tensor([1]) / loss.item())
            recon = recon.squeeze().cpu().numpy()
            imgN = imgN.squeeze().cpu().numpy()
            result = cv2.hconcat((imgN, recon))

            cv2.imwrite(args.output_path + 'img_' + str(k) + '.jpg', (result*255).astype('uint8'))
            print('Loss: {:.2E}, PSNR: {:,.2f}, time: {:,.2f}'.format(Decimal(loss.item()), psnr.item(), recon_time))
