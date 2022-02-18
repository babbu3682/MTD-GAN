import SimpleITK as sitk 
import os
import pandas as pd
import pydicom
from option.PD_default import TestOptions
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def scale(x, out_range=(-1, 1)):
   domain = np.min(x), np.max(x)
   y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
   return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


opt = TestOptions()
path = '/repo3/Compressed_Sensing_and_Black_BLood/'
patient_num = [i + 1 for i in range(10)]
#patient_num = ['11','12','15','16']
#patient_num[-2] = 15; patient_num[-1] = 16
patient_num  = [str(item).zfill(2) for item in patient_num]

patient_path = os.listdir(path)
patient_id = [item.split(' ')[0][:2] for item in patient_path]

patient_path = pd.Series(patient_path)

for name in patient_num:
    try:
        os.makedirs('./test_result/' + opt.name + '/' + 'patient_' + name)
    except:
        pass
    try:
        os.makedirs('./test_result/' + opt.name + '/' + 'patient_' + name+'/input')
        os.makedirs('./test_result/' + opt.name + '/' + 'patient_' + name+'/target')
        os.makedirs('./test_result/' + opt.name + '/' + 'patient_' + name+'/output')    
    except:
        pass

for item in patient_num:
    idx = [s == item for s in patient_id]
    path1 = path + list(patient_path[idx])[0]
    path1 = path1 + '/PD/original/'
    path1 = path1 + os.listdir(path1)[0]
    
    filename = os.listdir(path1)
    zpositions = []
    ###MAKE AXIAL LIST###
    for i in range(len(filename)):
        dcm =  pydicom.read_file(path1 + '/' + filename[i])
        zpositions.append(dcm.ImagePositionPatient[-1])
    
    df = pd.DataFrame({'filename' : filename, 'zposition' : zpositions})
    df = df.sort_values('zposition')
    df = df.reset_index(drop = True)
    path_np = './test_result/' + opt.name + '/Img_array/'
    path_nps = [ss for ss in os.listdir(path_np) if ss.split('_')[0] == item]
    path_nps.sort()

    name11= path_nps[0].split('_')[0] + '_'+path_nps[0].split('_')[1]

    for i in range(len(filename)):
        dcm1 = pydicom.read_file(path1+'/'+df['filename'][i])
        mins ,maxs = dcm1.pixel_array.min() , dcm1.pixel_array.max()
        inputs ,target , output = np.load(path_np + name11 + '_' + str(len(filename) - i - 1) + '.npy')        
        inputs = scale(inputs, out_range = (mins,maxs)); target = scale(target, out_range = (mins,maxs))
        output = scale(output , out_range = (mins,maxs))


        inputs = inputs.astype(np.uint16); target = target.astype(np.uint16); output = output.astype(np.uint16)

        dcm1.PixelData = np.flipud(inputs[0,:]).tobytes()
        dcm1.save_as('./test_result/' + opt.name + '/patient_'+item+'/input/' + name11 + '_' + str(len(filename) - i)  + '_input.dcm')

        dcm1.PixelData = np.flipud(output[0,:]).tobytes()
        dcm1.save_as('./test_result/' + opt.name + '/patient_'+item+'/output/' + name11 + '_' + str(len(filename) - i)  + '_output.dcm')

        dcm1.PixelData = np.flipud(target[0,:]).tobytes()
        dcm1.save_as('./test_result/' + opt.name + '/patient_'+item+'/target/' + name11 + '_' + str(len(filename) - i)  + '_target.dcm')
    
