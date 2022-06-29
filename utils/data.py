import torch
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms, eval = None, use_cache=False):
        self.dataframe = dataframe
        self.transforms = transforms
        self.eval = eval
        self.cached_data = []
        self.use_cache = use_cache

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        if not self.use_cache:
            row = self.dataframe.iloc[index] 
            imClean = self.transforms(Image.open(row['pClean']))
            imNoisy = self.transforms(Image.open(row['pNoisy'])) 
            # patches 
            if self.eval == None: 
                imClean = imClean.unfold(1, 40, 10).unfold(2, 40, 10)
                imNoisy = imNoisy.unfold(1, 40, 10).unfold(2, 40, 10)
                imClean = imClean.resize_(1,imClean.shape[1]+imClean.shape[2],40,40)
                imNoisy = imNoisy.resize_(1,imNoisy.shape[1]+imNoisy.shape[2],40,40)
                for i in range(imClean.shape[1]): 
                    imC = imClean[:,i,:,:]
                    imN = imNoisy[:,i,:,:] 
                    if (imN == 1.0).sum() < imN.size(1)**2:   # No guardar patches blancos!     
                        self.cached_data.append((imC,imN))
            else:
                self.cached_data.append((imClean,imNoisy))
                return imClean,imNoisy
        else:
            imC, imN = self.cached_data[index]
        return imC, imN

    def set_use_cache(self, use_cache):
        if not use_cache:
            self.cached_data = []
        self.use_cache = use_cache
    
    def __len__(self):
        return len(self.dataframe)


