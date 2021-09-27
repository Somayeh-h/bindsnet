import os
import pickle
import numpy as np
from struct import unpack
import torch
from tqdm import tqdm
import math
import cv2
import matplotlib.pyplot as plt 
from bindsnet.encoding import PoissonEncoder
from PIL import Image
from bindsnet.encoding.encodings import poisson


def loadImg(imPath, reso=None):
    """
    imPath: full image path
    reso: (width,height)
    """
    # read and convert image from BGR to RGB 
    im = cv2.imread(imPath)[:,:,::-1]
    if reso is not None:
        im = cv2.resize(im,reso)
    return im


def plot_img(img, path):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.savefig(path)


def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)


def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out


def processImage(img, ftType, imWidth, imHeight, num_patches, i):

    img = cv2.resize(img,(imWidth, imHeight))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # if i == 0: 
    #     plot_img(img, 'Org gray')

    if ftType == "downsampled_patchNorm":
        im_norm = patch_normalise_pad(img, num_patches) 

        # Scale element values to be between 0 - 255
        img = np.uint8(255.0 * (1 + im_norm) / 2.0)

    return img


def processImageDataset(path, type, imWidth, imHeight, time=350, dt=1, ftType="downsampled_patchNorm", num_patches=7, limit=math.inf, skip_ratio=8, offset=0, offset_after_skip=0, num_multi_imgs=1, desired_num_labels=50, place_familiarity=False, intensity=4):

    print("Computing features: {} for image path: {} ...\n".format(ftType, path))

    imgLists = []
    input_intensity = 2

    for p in path: 
        imgList = np.sort(os.listdir(p))
        imgList = [os.path.join(p,f) for f in imgList]    
        imgLists.append(imgList)

    frames = []
    org_frames = [] 
    encoded_frames = [] 

    nordlandPath = "/nordland/" 
    nordland = False
    content = []
    if nordlandPath in path[0]:
        nordland = True
        # load text file 
        dirPath = os.path.abspath(os.getcwd())
        filtered_names = dirPath + "/nordland_imageNames.txt"

        with open(filtered_names) as f:
            content = f.read().splitlines()
        
        filtered_index = [int(''.join(x for x in char if x.isdigit())) for char in content] 

    paths_used = [] 

    for repeat in range(num_multi_imgs): 
        imgListCount = 0 
        for imgList in imgLists: 
            mean_total_brightness = []
            if nordlandPath in imgList[0]: 
                nordland = True
                skip = int(len(imgList) * skip_ratio) if skip_ratio < 1 else skip_ratio
            else:
                nordland = False 
                skip = int(len(imgList) * skip_ratio) if skip_ratio < 1 else skip_ratio
            j = 0
            ii = 0 
            kk = 0

            for i, imPath in enumerate(imgList):
                
                # test for place familiarity - for last imgList, pick the imgs that are not used in inference
                if place_familiarity and type == "test" and imgListCount == len(imgLists)-1 and (i == 0 or i < 800 ):
                    skip = 5
                    continue

                if i == limit:
                    break
                
                if ".jpg" not in imPath and ".png" not in imPath:
                    i += 1
                    continue 
                
                if nordland: 
                    if (i not in filtered_index) or (( repeat == 0 and num_multi_imgs > 1) and ( (i-1) not in filtered_index) ):
                        continue

                    if j % skip != 0 or (skip <= 10 and j < offset):
                        j += 1
                        continue
                    j += 1
                
                if not nordland and ( (skip != 0 and i % skip != 0) or i < offset):  
                    continue
                
                if ( ii != 0 and ii % desired_num_labels == 0 ):
                    break 

                if (offset_after_skip > 0 and kk < offset_after_skip):
                    kk += 1
                    continue
                
                kk += 1 
                index = i

                if ( repeat == 0 and num_multi_imgs > 1) or type == "val": 
                    if i == 0 or nordland:
                        index = i+1
                    else: 
                        index = i-1

                    # index = i-1
                    imPath = imgList[index]

                elif repeat == 2:
                    if i == 0:
                        index = i+2
                    else: 
                        index = i+1

                    # index = i+1
                    imPath = imgList[index]

                frame = loadImg(imPath)

                if nordland: 
                    idx = np.where(np.array(filtered_index) == index)[0][0]
                    path_id = filtered_index[idx]
                else:
                    path_id = index

                # path_id = i
                paths_used.append(path_id)

                if num_multi_imgs == 1 or repeat == 1: 
                    # if not nordland: 
                    frame = cv2.resize(frame,(640, 360))
                    org_frames.append(frame)

                # if ii == 0 and (num_multi_imgs == 1 or repeat == 1):
                    # plot_img(frame, 'Org img ' + type + "_" + str(ii) + "_" + str(i) )

                frame = processImage(frame,ftType, imWidth, imHeight, num_patches, i)  
                frames.append(frame)

                flattened_frame = frame.reshape(imWidth*imHeight) / intensity         # 8. * input_intensity

                flattened_frame = flattened_frame.astype(np.uint8)

                encoded_frame = poisson(torch.from_numpy(flattened_frame), time, dt, device="cpu")

                encoded_frame = encoded_frame.view([1, 350, 1, 28, 28])

                encoded_frames.append(encoded_frame)

                mean_brightness = np.array(frame).mean(axis=0)
                mean_brightness = mean_brightness.mean()

                mean_total_brightness.append(mean_brightness)

                # if ii == 0 and (num_multi_imgs == 1 or repeat == 1):
                #     plot_img(frame, type + "_" + str(ii) + "_" + str(i))   

                ii += 1
            
            imgListCount += 1 
            
            print("mean brightness:\n{} in path\n{}".format(mean_total_brightness, imgList[0]))

    frames = np.array(frames)

    if np.any(np.array(paths_used)):
        print("indices used in {}:\n{}".format(type, paths_used))

    np.save('outputs/org_{}_frames.npy'.format(type), org_frames)
    np.save('outputs/{}_frames.npy'.format(type), frames)

    return frames, encoded_frames


