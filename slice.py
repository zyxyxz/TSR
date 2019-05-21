import os
import nibabel as nib
import scipy.misc
import numpy as np

FCD_ROOT_PATH = os.path.abspath("/home/henry/ai/dlDate/FCD/")


# 按需求切片
def save_slice(img_slice, label_slice, floder, key, i):
        img_path = os.path.join("../../../../dlDate/all_fcd/train", folder)
        label_path = os.path.join("../../../../dlDate/all_fcd/train_ann", folder)
        img_name = floder + '_' + key + '_' + str(i) + '.jpg'
        label_name = floder + '_label' + '_' + key + '_' + str(i) + '.jpg'
        scipy.misc.imsave(img_path + '/' + img_name, img_slice)
        scipy.misc.imsave(label_path + '/' + label_name, label_slice)


def mkdir(folder):
    train_path = os.path.join("../../../../dlDate/all_fcd/train", folder)
    train_ann_path = os.path.join("../../../../dlDate/all_fcd/train_ann", folder)
    Folder = os.path.exists(train_path)
    if not Folder:
        os.makedirs(train_path)
        os.makedirs(train_ann_path)
        print("OK")
    else:
        print("exists!")


def load_data(file):
    FILE_PATH = os.path.join(FOLDER_PATH, file)

    data = nib.load(FILE_PATH)
    data_arr = data.get_fdata()
    data_arr = np.squeeze(data_arr)
    return data_arr


if __name__ == '__main__':
    for i in range(1, 12):
        if (i < 10):
            folder = 'w0' + str(i)
        else:
            folder = 'w' + str(i)
        FOLDER_PATH = os.path.join(FCD_ROOT_PATH, folder)

        img_file = folder + '_brain.nii'
        label_file = folder + '_brain-label.nii'

        # mkdir(folder)

        img = load_data(img_file)
        label = load_data(label_file)

        dict = {'R': 157, 'Y': 189, 'G': 156}
        for key in dict:
            if key == 'R':
                for i in range(dict[key]):
                    img_slice = np.squeeze(img[i, :, :])
                    label_slice = np.squeeze(label[i, :, :])
                    save_slice(img_slice, label_slice, folder, key, i)
            elif key == 'Y':
                for i in range(dict[key]):
                    img_slice = np.squeeze(img[:, i, :])
                    label_slice = np.squeeze(label[:, i, :])
                    save_slice(img_slice, label_slice, folder, key, i)
            else:
                for i in range(dict[key]):
                    img_slice = np.squeeze(img[:, :, i])
                    label_slice = np.squeeze(label[:, :, i])
                    save_slice(img_slice, label_slice, folder, key, i)