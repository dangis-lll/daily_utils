import os
import SimpleITK as sitk

if __name__ == '__main__':

    dicom_path = r'C:\DL_DataBase\CBCT_data\raw_data\new\Dicom'
    outputpath = r'CBCT_data/raw_data/img'

    datalist = os.listdir(dicom_path)
    for i in datalist:
        try:
            # filename = os.listdir(os.path.join(dicom_path,i,'CBCT'))[0]
            # filepath = os.path.join(os.path.join(dicom_path,i,'CBCT'),filename)
            filepath = os.path.join(dicom_path, i)
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
            nb_series = len(series_IDs)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_IDs[0])
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            image3D = series_reader.Execute()
            sitk.WriteImage(image3D, os.path.join(dicom_path, '{}.nii.gz'.format(i)))
        except:
            print(i)

# filepath = r'CBCT_data/raw_data/1.3.6.1.4.1.30071.6.48549623791460.6530219950380206.1/1.3.6.1.4.1.30071.6.48549623791460.6530219950380206.1'
# series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
# print(series_IDs)
# series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_IDs[0])
# series_reader = sitk.ImageSeriesReader()
# series_reader.SetFileNames(series_file_names)
# image3D = series_reader.Execute()
# sitk.WriteImage(image3D, os.path.join(r'C:\DL_DataBase\CBCT_data\raw_data\1.3.6.1.4.1.30071.6.48549623791460.6530219950380206.1','bosstest.nii.gz'))
