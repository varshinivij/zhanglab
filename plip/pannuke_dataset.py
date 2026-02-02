import sys, os, platform, copy
import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import multiprocess as mp
from tqdm import tqdm
from PIL import Image, ImageFile
opj=os.path.join
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_PanNuke(root_dir, seed=None, train_ratio=None):
    df = pd.read_csv('{Path to PanNuke dataset}/processed_threshold=10_0.3/PanNuke_all_binary.csv',index_col=0)
    df = df.reset_index(drop=True)
    for i in df.index:
        caption = df.loc[i, 'caption']
        if 'malignant' in caption:
            tissue = caption.split('malignant ')[1].split(' tissue')[0]
            df.loc[i, 'tissue'] = tissue
            df.loc[i, 'label'] = 1
            df.loc[i, 'label_text'] = 'malignant'
            df.loc[i, 'label_tissue'] = 'malignant %s' % tissue
            df.loc[i, 'caption_no_tissue'] = caption.replace(tissue + ' ', '')
        elif 'benign' in caption:
            tissue = caption.split('benign ')[1].split(' tissue')[0]
            df.loc[i, 'tissue'] = tissue
            df.loc[i, 'label'] = 0
            df.loc[i, 'label_text'] = 'benign'
            df.loc[i, 'label_tissue'] = 'benign %s' % tissue
            df.loc[i, 'caption_no_tissue'] = caption.replace(tissue + ' ', '')
        else:
            print(caption)
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    uniq_tissuetypes = df['tissue'].unique()
    

    # equally split dataset into train and test for each cancer subtype and each label
    train = pd.DataFrame()
    test = pd.DataFrame()
    for tissue in uniq_tissuetypes:
        for label_text in ['benign', 'malignant']:
            df_subset = df.loc[(df['tissue'] == tissue) & (df['label_text'] == label_text)]

            # shuffle data
            df_subset = df_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
            # randomly split data into training and testing.
            df_subset_train = df_subset.iloc[:int(len(df_subset)*train_ratio),:].reset_index(drop=True)
            df_subset_test = df_subset.iloc[int(len(df_subset)*train_ratio):,:].reset_index(drop=True)

            train = pd.concat([train, df_subset_train], axis=0)
            test = pd.concat([test, df_subset_test], axis=0)
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    train = train[['image', 'label', 'label_text', 'label_tissue', 'caption', 'caption_no_tissue']]
    train.columns = ['image', 'label', 'label_text', 'text_style_0', 'text_style_1', 'text_style_4']

    test = test[['image', 'label', 'label_text', 'label_tissue', 'caption', 'caption_no_tissue']]
    test.columns = ['image', 'label', 'label_text', 'text_style_0', 'text_style_1', 'text_style_4']

    return train, test


def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def resizeimg(fp, this_savedir):
    pbar.update(mp.cpu_count())
    newsize = 224
    img = Image.open(fp)
    filename = os.path.basename(fp)
    if img.size[0] != img.size[1]:
        width, height = img.size
        min_dimension = min(width, height) # Determine the smallest dimension
        scale_factor = newsize / min_dimension # Calculate the scale factor needed to make the smallest dimension 224
        # Calculate the new size of the image
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height)) # Resize the image using the calculated size
        # center crop
        left = (width - newsize) / 2 # Calculate the coordinates to crop the center of the image
        top = (height - newsize) / 2
        right = left + newsize
        bottom = top + newsize
        img_resize = img.crop((left, top, right, bottom)) # Crop the image using the calculated coordinates
    else:
        img_resize = img.resize((newsize, newsize))
    new_savename = opj(this_savedir, filename)
    img_resize.save(new_savename)
    return new_savename

if __name__ == '__main__':

    seed=1
    train_ratio=0.7

    img_savedir = "/Users/varshinivijay/zhanglab/plip/data_validation_images_resize=224"
    savedir = "/Users/varshinivijay/zhanglab/plip/dataset/trainratio=0.7_size=224"
    os.makedirs(img_savedir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    root_dir = "/Users/varshinivijay/zhanglab/plip"

    
    #############################################
    #    PanNuke
    #############################################
    print('Processing PanNuke (normal, abnormal) dataset ...')
    train, test = process_PanNuke(root_dir, seed=seed, train_ratio=train_ratio)
    this_savedir_train = opj(img_savedir, 'PanNuke', 'train')
    this_savedir_test = opj(img_savedir, 'PanNuke', 'test')
    os.makedirs(this_savedir_train, exist_ok=True)
    os.makedirs(this_savedir_test, exist_ok=True)
    
    pbar = tqdm(total=int(len(train)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_train), X = train['image'])
    train['image'] = new_image_paths
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    train.to_csv(opj(savedir, 'PanNuke_train.csv'))
    test.to_csv(opj(savedir, 'PanNuke_test.csv'))
    