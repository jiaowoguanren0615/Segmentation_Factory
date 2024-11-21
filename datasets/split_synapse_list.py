import glob
import os


def write_name():

    if not os.path.exists('../lists/lists_Synapse'):
        os.makedirs('../lists/lists_Synapse', exist_ok=True)

    # npz files path
    files = glob.glob(r'/mnt/f/Synapse/Synapse/train_npz/*.npz')
    # files = glob.glob(r'/mnt/f/Synapse/Synapse/test_vol_h5/*.h5')
    # txt文件路径
    f = open(r'../lists/lists_Synapse/train.txt', 'w')
    # f = open(r'../lists/lists_Synapse/test_vol.txt', 'w')

    for i in files:
        name = i.split('\\')[-1]

        # TODO: when you generate train.txt
        name = name[:-4] + '\n'

        # name = name[:-7] + '\n'
        f.write(name)

    print("Finished!")


write_name()