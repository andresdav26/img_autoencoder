from torchvision import transforms
import math

def padd(Img,d):
    r = Img.shape[1] # rows
    c = Img.shape[2] # cols
    padr = math.ceil(r/d)*d -r 
    padc = math.ceil(c/d)*d -c
    pad_f = transforms.Pad([0,0,padc,padr], fill=1) 
    imNoisy_padd = pad_f(Img)
    return imNoisy_padd, r, c, padr, padc