from PIL import Image
from itertools import product

import numpy as np
import os 

def tile(src, pi, dir_in, dir_out, d):

    i0 = pi[0]
    i1 = pi[1]
    i2 = pi[2] 
    i3 = pi[3]
    i4 = pi[4]

    wfile = 1
    dir_list = os.listdir(src)
    for filename in dir_list:
        if filename.endswith(".png"):
            print(filename)
            _, ext = os.path.splitext(filename)
            img_N = Image.open(os.path.join(dir_in, 'noisy', filename)).convert('L')
            img_C = Image.open(os.path.join(dir_in, 'clean', filename)).convert('L')

            img_N = img_N.resize((1920,1920), Image.ANTIALIAS)
            img_C = img_C.resize((1920,1920), Image.ANTIALIAS)
            w, h = img_N.size
            print(w,h)
    
            grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
            for i, j in grid:

                if int(f'{i0}{i1}{i2}{i3}{i4}') <= 2501:
                    tp = 'train/'
                else: 
                    tp = 'test/'

                box = (j, i, j+d, i+d)
                out_N = os.path.join(dir_out, tp, 'noisy', f'img_{i0}{i1}{i2}{i3}{i4}{ext}')
                out_C = os.path.join(dir_out, tp, 'clean', f'img_{i0}{i1}{i2}{i3}{i4}{ext}')
        
                if ((np.asarray(img_N.crop(box)) == 255)*1).sum() < img_N.crop(box).size[0]**2:
                    img_N.crop(box).save(out_N)
                    img_C.crop(box).save(out_C)

                    i4 = i4 + 1
                    if i4 > 9:
                        i4 = 0
                        i3 = i3 +1 
                        if i3 > 9:
                            i3 = 0
                            i2 = i2 + 1
                            if i2 > 9: 
                                i2 = 0
                                i1 = i1 + 1
                                if i1 > 9: 
                                    i1 = 0
                                    i0 = i0 + 1
                else: 
                    wfile = wfile + 1
    print(wfile)
# if __name__ == '__main__': 

#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--dir_input", required=True, help="Input image(s) path(s)")
#     ap.add_argument("-o", "--dir_output", required=True, help="outPut image")
#     ap.add_argument("-f", "--filename", required=True, help = "file name")
#     args = vars(ap.parse_args())

src = '/home/adguerrero/ia_nas/datasets/autoencoder/dataset/datos_andres/noisy/'
dir_input = '/home/adguerrero/ia_nas/datasets/autoencoder/dataset/datos_andres/'
dir_output = '/home/adguerrero/ia_nas/datasets/autoencoder/dataset/datos_andres/crop/'
tile(src,[0,0,0,0,0], dir_input, dir_output, 128)