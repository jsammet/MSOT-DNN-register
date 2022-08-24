import os
import torch
import torchio as tio
import numpy
from pathlib import Path

def reorder_axis(image):
    array = image.tensor
    
    array = array.flip(1)
    image = tio.ScalarImage(tensor=array)
    return image
 
def reader(inshape_3D, mri_path, mask_path, input_path, label_path):
  msot_data = []
  msot_label = []
  mri_data = []
  file_list = [] 
  rescale = tio.RescaleIntensity(out_min_max=(0.0, 1.0))
  reorder = tio.ToCanonical()
  ranged = tio.transforms.Clamp(-1e1)
  normalize = tio.ZNormalization()
  transform = tio.CropOrPad(inshape_3D)
  #transform = tio.transforms.Resample((2,2,1))

  #Reference MRI iamge
  mri_subject = tio.ScalarImage(Path(mri_path))
  print("MRI Image: ",mri_subject.shape)
  print("MRI:")
  print("Intensity max: ", torch.max(mri_subject.data))
  print("Intensity min: ", torch.min(mri_subject.data))
  print("MRI Spacing: ",mri_subject.spacing)
  mri_subject = rescale(mri_subject)
  mri_subject = transform(mri_subject)
  mri_subject.plot()
  mri_data.append(mri_subject) 

  mask_AOI = tio.ScalarImage(Path(mask_path))
  print("Mask:")
  print("Mask Image: ",mask_AOI.shape)
  print("Intensity max: ", torch.max(mask_AOI.data))
  print("Intensity min: ", torch.min(mask_AOI.data))
  print("Mask Spacing: ",mask_AOI.spacing)
  mask_AOI = reorder_axis(mask_AOI)
  mask_AOI = rescale(mask_AOI)
  mask_AOI = transform(mask_AOI)
  mask_AOI.plot()

  img_size = []
  label_set = []
  label_check = None
  image_check = None
  '''
  This part will have to be altered to account for changes in naming and/or folder sturcture!
  '''
  for dirname, _, filenames in os.walk(input_path):
      for filename in filenames:
          label_path_1 = Path(label_path + filename)
          if filename.startswith('transformed'):
              label_start = filename.split(' ')[1]
              label_name = label_start.split('min')[0] + 'new.nii'
              label_path_2 = Path(label_path + label_name)
              if label_path_2.is_file():
                  label_path = label_path_2
              else:
                  continue
          else:
              continue
          print(label_path)
          file_list.append(label_name)
          label_check = label_path
          label_subject = tio.ScalarImage(Path(label_path))
          label_sz = label_subject.shape
          label_set.append((label_sz[1],label_sz[2],label_sz[3]))
          label_subject = rescale(label_subject)
          label_subject = transform(label_subject)
          msot_label.append(label_subject)

          image_path = os.path.join(dirname, filename)
          image_check = image_path
          img_subject = tio.ScalarImage(Path(image_path))
          img_sz = img_subject.shape
          img_subject = reorder_axis(img_subject)
          img_size.append((img_sz[1],img_sz[2],img_sz[3]))
          img_subject = rescale(img_subject)
          img_subject = transform(img_subject)
          msot_data.append(img_subject)

  print('MSOT Dataset size:', len(msot_data), 'subjects')
  print('MSOT label set size:', len(msot_label), 'subjects')
  print('MRI Dataset size:', len(mri_data), 'subjects')

  print("dataset image 1 shape: ", msot_data[0])
  print("label image 1 shape: ", msot_label[0])
  print("Reference image shape: ", mri_data[0])
