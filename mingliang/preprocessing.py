import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

MIN_BOUND = 0
MAX_BOUND = 35
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

if __name__ == '__main__':
    no_slicies = 10

    first_patient = load_scan(INPUT_FOLDER + patients[0])
    first_patient_pixels = get_pixels_hu(first_patient)
    first_patient_pixels = normalize(first_patient_pixels)
    length_of_stack = int(len(first_patient_pixels)/no_slicies)
    reminder = len(first_patient_pixels)%no_slicies
    sgements = [[first_patient_pixels[index:index+length_of_stack]]for index in range(0,len(first_patient_pixels)-length_of_stack,length_of_stack)]
    sgements[-1].append(first_patient_pixels[len(first_patient_pixels)-reminder:len(first_patient_pixels)])
    
    for sgement in sgements:
        arr = np.zeros((512, 512), np.int16)
        count = 3
        for im in sgement:
            smallest = np.amin(im)
            biggest = np.amax(im)
            
            #imarr = np.array(im, dtype=np.int16)        
            arr = arr + (1 - im) * np.log(count)/(biggest - smallest)

            #print ((N * 14)/ np.log10(count))
            count = count + 1
            #arr = np.array(np.round(arr), dtype=np.uint8)
            arr = np.array(np.round(arr),dtype=np.uint8)
        #out=Image.fromarray(arr, mode='L')
        plt.imshow(arr[0], cmap=plt.cm.gray)
        plt.show()
        

