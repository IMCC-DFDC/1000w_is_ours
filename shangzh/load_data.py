from torchvision.datasets import DatasetFolder
import os
import json
import torch.utils.data as data

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Dataset(DatasetFolder):
    def __init__(self, root, phase, loader=default_loader, transform=None, target_transform=None):
        path_json = '/home/shangzh/DFDC/code/dataset.json'
        f = open(path_json)
        dict = json.load(f)
        train_files = []
        test_files = []
        for i in dict:
            if dict[i]['set'] == 'train':
                train_files.append(i.split('/')[-1][0:-4])
            elif dict[i]['set'] == 'test':
                test_files.append(i.split('/')[-1][0:-4])
        if phase == 'train':
            self.files_set = train_files
        elif phase == 'test':
            self.files_set = test_files
        self.samples = []
        self.num = 0
        with open(root) as f:
            lines = f.readline()
            while lines!='':
                dir, label = lines.split()
                if not dir.split('/')[-1] in self.files_set:
                    lines = f.readline()
                    continue
                for root, dirs, files in os.walk(dir):
                    for file in files:
                        self.samples.append((os.path.join(root,file),int(label)))
                lines = f.readline()
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform


# f = open('dataset.json')
# dict = json.load(f)
# train_files = []
# test_files = []
# for i in dict:
#     if dict[i]['set'] == 'train':
#         train_files.append(i.split('/')[-1][0:-4])
#     elif dict[i]['set'] == 'test':
#         test_files.append(i.split('/')[-1][0:-4])


