import numpy as np

def OpenFWI(base_dir, instance_size, in_channels, split, start):
    file_num = instance_size // 500

    print("data_dir: ", base_dir)
    for i in range(start, start + file_num):
        temp_seismic_file = base_dir + '{}_data/seismic/seismic{}.npy'.format(split, i)
        temp_vmodel_file = base_dir + '{}_data/vmodel/vmodel{}.npy'.format(split, i)
        # 打印周期为2
        if (i + 1) % 2 == 0: print("Read the {}-th file~~~~".format(i+1))

        _data_set = np.load(temp_seismic_file)
        _label_set = np.load(temp_vmodel_file)

        if i == start:
            data_set = _data_set
            label_set = _label_set
        else:
            data_set = np.append(data_set, _data_set, axis=0)
            label_set = np.append(label_set, _label_set, axis=0)

    return data_set, label_set