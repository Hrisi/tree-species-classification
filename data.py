import os
import glob
#import h5py
import numpy as np
from torch.utils.data import Dataset
import pickle
import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def _load_data_file(name):
    print('Load processed data from %s...' % name)
    with open(name, 'rb') as f:
        data, label = pickle.load(f)
    return data, label

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def load_data(partition):
    #download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    FILE = os.path.join(DATA_DIR, 'modelnet40_%s_4098pts_fps.dat' % partition)
    all_data = []
    all_label = []
    all_label_int = []
    all_data, all_label = _load_data_file(FILE)
    all_data = np.array(all_data)
    all_label = np.array(all_label, dtype=np.int64)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data = []
        data_tmp, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

        for i in range(len(data_tmp)):
            if len(self.data_tmp) > num_points:
                self.data.append(farthest_point_sample(data_tmp[i], self.num_points))
            else:
                self.data.append(data_tmp[i])
        self.data = np.array(self.data)


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# Ref: https://github.com/jiachens/ModelNet40-C/blob/795672578f41d11ed5169f8f0ef259f7f441b6e6/dataloader.py#L157
class ModelNet40C(Dataset):
    def __init__(self, num_points, corruption, severity=None):
        super(ModelNet40C, self).__init__()

        assert corruption in [
            "background", "cutout", "density", "density_inc", "distortion",
            "distortion_rbf", "distortion_rbf_inv", "gaussian", "impluse", "lidar",
            "occlusion", "rotation", "shear", "uniform", "upsampling", # 15 corruptions
            "original",
        ]

        if corruption == "original":
            assert severity is None
            fname = "data/modelnet40c/data_original.npy"
        else:
            assert severity is not None
            fname = f"data/modelnet40c/data_{corruption}_{severity}.npy"

        self.data = np.load(fname)
        self.label = np.load("data/modelnet40c/label.npy")
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item]
        if pointcloud.shape[1] >= self.num_points:
            pointcloud = pointcloud[:self.num_points]
        else:
            choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
            pointcloud = pointcloud[choice, :]

        label = self.label[item]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
