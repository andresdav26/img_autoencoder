import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import Autoencoder 
from utils import data
from utils.dataframe import DF 

from PIL import Image
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
            transforms.Resize((1920,1920)),
            transforms.ToTensor()
            ])
            
# Current Device 
device = 'cpu'
print('Using device:', device)

# Define model
model = Autoencoder().to(device)

# load model
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.MSELoss()
d = 80 # size patch 
i = 0
j = 0
reconT = torch.zeros(1,1,1920,1920).to(device) 
with torch.no_grad():
        model.eval()
        start_time = time.time()
        imClean = transform(Image.open(args.baseroot + "clean/4401-16.png")).to(device)
        imNoisy = transform(Image.open(args.baseroot + "noisy/4401-16.png")).to(device)
        # patches 
        patchN = imNoisy.unfold(1, d, d).unfold(2, d, d)

        for i in range(patchN.shape[1]): 
            for j in range(patchN.shape[2]):
                imgN_p = patchN[:,i,j,:,:].unsqueeze(0)
                recon = model(imgN_p,device)
                reconT[:,:,i*d:(i+1)*d,j*d:(j+1)*d] = recon
        # Metrics 
        loss = criterion(reconT,imClean.unsqueeze(0))
        psnr = 10 * torch.log10(torch.tensor([1]) / loss.item())
        reconT = reconT.squeeze().cpu().numpy()    
        imNoisy = imNoisy.squeeze().cpu().numpy()
        result = cv2.hconcat((imNoisy, reconT))
        recon_time = time.time() - start_time
            
        cv2.imwrite(args.output_path + 'img_' + '.jpg', (result*255).astype('uint8'))
        print('Loss: {:.2E}, PSNR: {:,.2f}, time: {:,.2f}'.format(Decimal(loss.item()), psnr.item(), recon_time))
