import torch
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, eval, use_cache=False):
        self.dataframe = dataframe
        self.transform = transform
        self.eval = eval 
        self.cached_data = []
        self.use_cache = use_cache

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        if not self.use_cache:
            row = self.dataframe.iloc[index] 
            imClean = Image.open(row['pClean'])
            imNoisy = Image.open(row['pNoisy'])

            if self.transform: 
                imClean = self.transform(imClean)
                imNoisy = self.transform(imNoisy) 
            
            ## patches #####################################################################
            if self.eval:
                patchC = imClean.unfold(1, 128, 128).unfold(2, 128, 128)
                patchN = imNoisy.unfold(1, 128, 128).unfold(2, 128, 128)
                # imClean = imClean.resize_(imClean.shape[1]+imClean.shape[2],1,128,128)
                # imNoisy = imNoisy.resize_(imNoisy.shape[1]+imNoisy.shape[2],1,128,128)
                for i in range(patchC.shape[1]): 
                    for j in range(patchC.shape[2]):
                        imClean = patchC[:,i,j,:,:]
                        imNoisy = patchN[:,i,j,:,:] 
                        self.cached_data.append((imClean,imNoisy))
                return self.cached_data
            ################################################################################    
            self.cached_data.append((imClean,imNoisy))
        else:
            imClean, imNoisy = self.cached_data[index]
        return imClean, imNoisy

    def set_use_cache(self, use_cache):
        if not use_cache:
            self.cached_data = []
        self.use_cache = use_cache
    
    def __len__(self):
        return len(self.dataframe)


