import time, os
from img2dcm import convert as img_convert
from msk2dcm import convert as msk_convert
from msk2dcm import convert_list as msk_list_convert


def nifti2dcm(input_nifti_dir, output_dicom_dir, struct_name, roi_list):
    head, tail = os.path.split(input_nifti_dir)
    patienName = tail
    patienID = time.strftime("%Y%m%d%H%M%S") + '_' + patienName
    img_path = os.path.join(input_nifti_dir, 'img.nii.gz')
    msk_path = os.path.join(input_nifti_dir, 'label.nii.gz')
    img_convert(patienID, patienName, img_path, output_dicom_dir)
    print('Image conversion completed. ')
    msk_convert(msk_path, output_dicom_dir, output_dicom_dir, struct_name, roi_list)
    print('Label conversion completed. ')


def nifti_multiple_label_convert(input_nifti_dir, output_dir, struct_name):
    head, tail = os.path.split(input_nifti_dir)
    patienName = tail
    patienID = time.strftime("%Y%m%d%H%M%S") + '_' + patienName
    ids_ = os.listdir(input_nifti_dir)
    img_path = None
    msk_fn_list = []
    for i, pid in enumerate(ids_):
        # print('pid',pid)
        if 'img' in pid:
            img_path = os.path.join(input_nifti_dir, pid)
        elif 'label' not in pid:
            fn = pid[:-7]
            msk_fn_list.append(fn)

    print(img_path)
    output_dicom_dir = os.path.join(output_dir, patienName)
    os.makedirs(output_dicom_dir, exist_ok=True)
    img_convert(patienID, patienName, img_path, output_dicom_dir)
    msk_list_convert(input_nifti_dir, output_dicom_dir, output_dicom_dir, struct_name, msk_fn_list)


if __name__ == '__main__':

    # labelpath = r'C:\DL_DataBase\CBCT_data\kk/30730_cut.nii.gz'
    # patienName = '30730_cut'
    #
    # outputdir = labelpath[:-7]
    # if not os.path.exists(outputdir):
    #     os.mkdir(outputdir)
    # patienID = time.strftime("%Y%m%d%H%M%S") + '_' + patienName
    #
    # img_convert(patienID, patienName, labelpath, outputdir)

    filepath = r'C:\DL_DataBase\ddd\re\1.0'
    datalist = os.listdir(filepath)
    if len(datalist):
        for data in datalist:
            filename = data
            labelpath = os.path.join(filepath, filename)
            patienName = data[:-7]

            outputdir = filepath + '/' + patienName
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
            patienID = time.strftime("%Y%m%d%H%M%S") + '_' + patienName

            img_convert(patienID, patienName, labelpath, outputdir)

    # branddir = 'statisc/raw'
    # outputdir = r'C:\DL_DataBase\dcmlabel'
    # if not os.path.exists(outputdir):
    #     os.mkdir(outputdir)
    # brandlist = os.listdir(branddir)
    # if len(brandlist):
    #     for brand in brandlist:
    #         filepath = os.path.join(branddir, brand, 'label')
    #         datalist = os.listdir(filepath)
    #         if len(datalist):
    #             for data in datalist:
    #                 filename = data
    #                 labelpath = os.path.join(filepath, filename)
    #                 patienName = data[:-7]
    #
    #                 outputdir = filepath+'/'+patienName
    #                 if not os.path.exists(outputdir):
    #                     os.mkdir(outputdir)
    #                 patienID = time.strftime("%Y%m%d%H%M%S") + '_' + patienName
    #
    #                 img_convert(patienID, patienName, labelpath, outputdir)
