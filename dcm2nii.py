import os
import SimpleITK as sitk

if __name__ == '__main__':

    dicom_path = r'C:\C++_program\drspro\3oxz'
    outputpath = r'C:\C++_program\drspro\3oxz'

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
            outpath = os.path.join(outputpath, '{}.nii.gz'.format(i))
            sitk.WriteImage(image3D, outpath)
        except:
            print(i)

# filepath = r'CBCT_data/raw_data/1.3.6.1.4.1.30071.6.48549623791460.6530219950380206.1/1.3.6.1.4.1.30071.6' \
#            r'.48549623791460.6530219950380206.1'
# series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
# print(series_IDs)
# series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_IDs[0])
# series_reader = sitk.ImageSeriesReader()
# series_reader.SetFileNames(series_file_names)
# image3D = series_reader.Execute()
# sitk.WriteImage(image3D, os.path.join(r'C:\DL_DataBase\CBCT_data\raw_data\1.3.6.1.4.1.30071.6.48549623791460'
#                                       r'.6530219950380206.1','bosstest.nii.gz'))
