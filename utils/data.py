import torch
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms, use_cache=False):
        self.dataframe = dataframe
        self.transforms = transforms
        self.cached_data = []
        self.use_cache = use_cache

    def __getitem__(self, index):
        Image.MAX_IMAGE_PIXELS = None
        if not self.use_cache:
            row = self.dataframe.iloc[index] 
            imClean = self.transforms(Image.open(row['pClean']))
            imNoisy = self.transforms(Image.open(row['pNoisy'])) 
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


