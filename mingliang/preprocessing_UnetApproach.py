import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, filters, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans

# Some constants 
INPUT_FOLDER = '/Users/mingliangang/lungCancer/downloads/lungCancer/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

def load_scan(path):
	print(path)
	slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
	for s in slices:
		s.SliceThickness = slice_thickness
	return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

if __name__ == '__main__':
    no_slicies = 10

    first_patient = load_scan(INPUT_FOLDER + patients[0])
    img = get_pixels_hu(first_patient)
    num_images = len(img)
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #move the underflow bins
    img[img==max]=mean
    img[img==min]=mean

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  

    #Why 4D array ??
    final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
    final_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)

    for i in range(num_images):
        eroded = morphology.erosion(thresh_img[i],np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        img = mask*img[i]
        new_mean = np.mean(img[mask>0])  
        new_std = np.std(img[mask>0])

        old_min = np.min(img)       # background color
        img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
        img = img-new_mean
        img = img/new_std

        final_images[i,0] = img
        final_masks[i,0] = mask
    
    rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
    test_i = int(0.2*num_images)
    np.save("trainImages.npy",final_images[rand_i[test_i:]])
    np.save("trainMasks.npy",final_masks[rand_i[test_i:]])
    np.save("testImages.npy",final_images[rand_i[:test_i]])
    np.save("testMasks.npy",final_masks[rand_i[:test_i]])
