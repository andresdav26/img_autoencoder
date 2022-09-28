import torch
import torch.nn as nn
from torchvision import transforms

from model import Autoencoder 
from utils.padding import padd

from pathlib import Path 
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
                    help='path to noisy image')
parser.add_argument('-c', '--clean_path', required=False,
                    help='path to clean image')
parser.add_argument('-o', '--output_path', required=False,
                    help='path where model files and training information will be placed')
args = parser.parse_args()


src_folder = Path(args.baseroot)
img_paths = [path for path in src_folder.glob('*') if path.suffix in ('.png', '.jpg', '.jpeg')]

# Load data 
batch_size = 1
transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Resize((1920,1920)),
            # transforms.GaussianBlur(5),
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
for img_path in img_paths:
    with torch.no_grad():
            model.eval()
            start_time = time.time()
            imNoisy = transform(Image.open(img_path).convert('1')).to(device)
            imNoisy_padd, r, c, padr, padc = padd(imNoisy,d) # padding
            patchN = imNoisy_padd.unfold(1, d, d).unfold(2, d, d)
            reconT = torch.zeros(1,1,r+padr,c+padc).to(device) 
            for i in range(patchN.shape[1]): 
                for j in range(patchN.shape[2]):
                    imgN_p = patchN[:,i,j,:,:].unsqueeze(0)
                    recon = model(imgN_p,device)
                    reconT[:,:,i*d:(i+1)*d,j*d:(j+1)*d] = recon 
            recon_time = time.time() - start_time
            reconT = reconT[:,:,0:r,0:c]
            if args.clean_path:
                # Metrics 
                imClean = transform(Image.open(args.clean_path + str(img_path.name)).convert('1')).to(device)
                loss = criterion(reconT,imClean.unsqueeze(0))
                psnr = 10 * torch.log10(torch.tensor([1]) / loss.item())
                
                reconT = reconT.squeeze().cpu().numpy()   
                imNoisy = imNoisy.squeeze().cpu().numpy()
                result = cv2.hconcat((imNoisy, reconT))
                cv2.imwrite(args.output_path +  str(img_path.name), (result*255).astype('uint8'))
                print('Loss: {:.2E}, PSNR: {:,.2f}, time: {:,.2f}'.format(Decimal(loss.item()), psnr.item(), recon_time))
            else:
                reconT = reconT.squeeze().cpu().numpy()   
                imNoisy = imNoisy.squeeze().cpu().numpy()
                result = cv2.hconcat((imNoisy, reconT))
                cv2.imwrite(args.output_path +  str(img_path.name), (reconT*255).astype('uint8'))
                print('Time: {:,.2f}'.format(recon_time))
