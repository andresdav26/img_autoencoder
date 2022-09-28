import pandas as pd 
import glob 
import os 

def DF(baseroot, type):

    # clean
    dfC = pd.DataFrame()
    if type == 'train': 
        dfC['pClean'] = glob.glob(baseroot + "train5/clean/*.png")
    elif type == 'test':
        dfC['pClean'] = glob.glob(baseroot + "test5/clean/*.png")
    elif type == 'val':
        dfC['pClean'] = glob.glob(baseroot + "val/clean/*.png")
    dfC['idC'] = [p.split(os.path.sep)[-1][-8:-4] for p in dfC['pClean']]
    dfC = dfC.sort_values(by=['idC'],ascending=[True])
    dfC.reset_index(inplace=True, drop=True)
    del dfC['idC']

    # Noisy
    dfN = pd.DataFrame()
    if type == 'train': 
        dfN['pNoisy'] = glob.glob(baseroot + "train5/noisy/*.png")
    elif type == 'test':
        dfN['pNoisy'] = glob.glob(baseroot + "test5/noisy/*.png")
    elif type == 'val':
        dfN['pNoisy'] = glob.glob(baseroot + "val/noisy/*.png")
    dfN['idN'] = [p.split(os.path.sep)[-1][-8:-4] for p in dfN['pNoisy']]
    dfN = dfN.sort_values(by=['idN'],ascending=[True])
    dfN.reset_index(inplace=True, drop=True)
    del dfN['idN']

    return pd.concat([dfC,dfN],axis=1)