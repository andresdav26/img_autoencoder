import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Autoencoder 
from utils import Data
from utils.DataFrame import DF 

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
transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),
            ])
val_dataset = Data.MyDataset(DF(args.baseroot,'val'), transform, 'val', use_cache=False)
valloader = DataLoader(val_dataset, batch_size=1)

# Current Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define model
model = Autoencoder().to(device)

# load model
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.MSELoss()

with torch.no_grad():
        model.eval()
        for i, pair in enumerate(valloader):
            start_time = time.time()
            imgC, imgN = pair[0].to(device), pair[1].to(device)
            recon = model(imgN)
            loss = criterion(recon,imgC)
            recon_time = time.time() - start_time
            
            # Peak Signal to Noise Ratio
            mse = torch.mean((recon - imgC) ** 2)
            psnr = 10 * torch.log10(1 / mse)

            recon = recon.squeeze().cpu().numpy()
            imgN = imgN.squeeze().cpu().numpy()
            result = cv2.hconcat((imgN, recon))
            cv2.imwrite(args.output_path + 'img_' + str(i) + '.jpg', (result*255).astype('uint8'))
            print('Loss: {:.4f}, PSNR: {:,.2f}, time: {:,.2f}'.format(loss.item(), psnr, recon_time))
