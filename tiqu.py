from __future__ import print_function
import os
import warnings

import pydicom

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor
import os
import SimpleITK as sitk
import warnings
from PIL import Image
from tqdm import tqdm

import pandas as pd
import io

basePath = r'D:\guke\data\1.1'
savePath = r'D:\guke\csv'
patient_list=os.listdir(basePath)
id=[]
save_file = np.empty(shape=[1,464])
#folders = os.listdir(basePath) # 读取featureExtraction文件夹下所有文件名
# print(folders)
df3 = pd.DataFrame()
df1 = pd.DataFrame()
# 将字符串转换成数据框
def str_to_dataframe(str_data):
    # 将字符串装换成字节流
    bytes_io = io.StringIO(str_data)

    # 使用pandas读取字节流并转换成Dataframe
    df = pd.read_csv(bytes_io)

    return df
def dicom_to_nii(dicom_file):
    # 读取DICOM文件
    dicom_data = pydicom.dcmread(dicom_file,force=True).pixel_array

    # 将像素值缩放到0-255的范围
    dicom_data_scaled = ((dicom_data - np.min(dicom_data)) / (np.max(dicom_data) - np.min(dicom_data)) * 255).astype(np.uint8)

    # 创建PIL Image对象
    image = sitk.GetImageFromArray(dicom_data_scaled)

    return image

for root,dirs,files in tqdm(os.walk(basePath)):

        for file in files:
            if ".dcm" in file and "AP" in file:
                dcm_path = os.path.join(root, file)
                mask_path = os.path.join(root, file.replace('AP.dcm', 'AP.nii'))
                # print(mask_path)
                # 读取XLSX表格文件
                # label_table_path = os.path.join(basePath, 'Label10.xlsx')
                # label = pd.read_excel(label_table_path)
                # print(label.info())
                #

                # imageFile = sitk.ReadImage(dcm_path,sitk.sitkInt8)
                imageFile=dicom_to_nii(dcm_path)
                # print(type(imageFile))
                # shape = (int(imageFile.Rows), int(imageFile.Columns), len(dicom_files))
                maskFile = sitk.ReadImage(mask_path)
                maskFile = sitk.Extract(maskFile, size=[maskFile.GetSize()[0], maskFile.GetSize()[1], 0], index=[0, 0, 0])


                # label_array = sitk.GetArrayFromImage(maskFile)
                # unique_labels = np.unique(label_array)
                # print(unique_labels)

                image_origin = imageFile.GetOrigin()
                image_spacing = imageFile.GetSpacing()

                mask_origin = maskFile.GetOrigin()
                mask_spacing=maskFile.GetSpacing()

                if image_origin !=mask_origin or image_spacing !=mask_spacing:
                    imageFile.SetOrigin(mask_origin)
                    imageFile.SetSpacing(mask_spacing)


                label_array=sitk.GetArrayFromImage(maskFile)
                unique_labels=np.unique(label_array)

                if 1 in unique_labels:

                    extractor1 = featureextractor.RadiomicsFeatureExtractor()
                    featureextractor1 = extractor1.execute(imageFile, maskFile, label=1)
                    df1_new = pd.DataFrame.from_dict(featureextractor1.values()).T
                    df1_new.columns = featureextractor1.keys()
                    df1 = pd.concat([df1, df1_new])
                    name=str_to_dataframe(mask_path)
                    df1 = df1.join(name)

                if 3 in unique_labels:
                    extractor3 = featureextractor.RadiomicsFeatureExtractor()
                    featureextractor3 = extractor3.execute(imageFile, maskFile, label=3)
                    df3_new = pd.DataFrame.from_dict(featureextractor3.values()).T
                    df3_new.columns = featureextractor3.keys()
                    df3 = pd.concat([df3, df3_new])
                    name = str_to_dataframe(mask_path)
                    df3 = df3.join(name)

# save_file = save_file.transpose()
                # print(save_file.shape)


df3.to_excel(os.path.join(basePath,'1.1APR.xlsx'))
df1.to_excel(os.path.join(basePath,'1.1APL.xlsx'))





