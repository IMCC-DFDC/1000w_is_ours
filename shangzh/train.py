import torch
import xception
import torchvision
import argparse
import os
import numpy as np
import time
from torch.autograd import Variable
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from load_data import Dataset
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', default=280, type=int,help='crop_size of frame')
parser.add_argument('--batch_size', default=32, type=int,help='batch_size')
parser.add_argument('-g', '--gpu', default=0, type=int,help='gpu id to be used')
parser.add_argument('-l', '--learnrate', default=0.1, type=float,
                    help='learningrate')
parser.add_argument('--weight-decay', default=0.0001, type=float,
                    help='parameter to decay weights')
parser.add_argument('-e', '--num_epochs', default=20, type=int,
                    help='number of epochs')
parser.add_argument('-m', '--model', default='xception', type=str,
                    help='which model~~~')
parser.add_argument('--phase', default='train', type=str,
                    help='path of model~~~')
args = parser.parse_args()

def main():
    save_path = '/home/shangzh/DFDC/model/'
    str_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
    print(str_time)
    save_path = save_path + '{}_{}_{}_'.format(args.model, str_time, args.learnrate)
    print(save_path)
    #############################################load dataset
    #data_root = '/home/shangzh/DFDC/dataset'
    data_train_root = '/home/shangzh/DFDC/code/dataset_1fps_crop_train(1).txt'
    data_test_root = '/home/shangzh/DFDC/code/dataset_1fps_crop_test(1).txt'
    transform_augment = T.Compose([
        T.Resize(args.crop_size),
        T.RandomHorizontalFlip(),
        #T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        T.RandomCrop(256, padding=0)
    ])
    transform_test = T.Compose([
        T.Resize(args.crop_size),
        T.CenterCrop(256),
    ])
    transform_normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=[0.47109917, 0.44989446, 0.40311126],
        #             std=[0.27472758, 0.26602452,
        #                  0.28146694])
    ])
    time1 = time.time()
    print(time1)
    train_dataset = Dataset(data_train_root, phase ='train',transform=T.Compose([transform_augment, transform_normalize]))
    time2 = time.time()
    print('load train set use {}s'.format(time2-time1))
    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)
    test_dataset = Dataset(data_test_root, phase ='test',transform=T.Compose([transform_test, transform_normalize]))
    loader_test = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=0,
                              pin_memory=True)
    #############################################load model
    model = xception.Xception(num_classes=3)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('xception-43020ad28.pth', map_location=lambda storage, loc: storage)
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and k[0:2] != 'fc':
            model_dict[k] = v
    model.load_state_dict(model_dict)
    model = model.cuda(args.gpu)
    ############################################ set up optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    fc_id = id(model.fc.parameters())
    optimizer = optim.SGD([{'params':filter(lambda p: id(p) == fc_id, model.parameters()),'lr':args.learnrate}],
                          momentum=0.9, weight_decay=args.weight_decay)
    ################################################## start training
    print('Training for %d epochs with learning rate %f' % (args.num_epochs, args.learnrate))
    best_acc = 0
    for epoch in range(args.num_epochs):
        if epoch >= 3:
            optimizer = optim.SGD([{'params':filter(lambda p: id(p) == fc_id, model.parameters()),'lr':args.learnrate},
                                   {'params': filter(lambda p: id(p) != fc_id, model.parameters()),
                                    'lr': 0.001}],
                          momentum=0.9, weight_decay=args.weight_decay)
        ############################## train phase
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs))
        if epoch == 0:
            time1 = time.time()
        if args.phase == 'train':
            train(loader_train, model, criterion, optimizer)
            torch.save(model.state_dict(), save_path + 'final.pkl')
            print('Final model save at {} epoch'.format(epoch + 1))
        if epoch == 0:
            time2 = time.time()
            print('train phase cost time:', time2 - time1)
        ############################## test phase
        # if args.phase == 'train' and (epoch+1) <=-1:
        #     continue
        print('testing test set')
        acc = check_accuracy(model, loader_test)
        if acc >= best_acc:
            if args.phase == 'train':
                torch.save(model.state_dict(), save_path + 'best.pkl')
                print('Best model save at {} epoch'.format(epoch + 1))
            best_acc = acc
        if epoch == 0:
            time3 = time.time()
            print('test cost time:', time3 - time2)
        if args.phase == 'test':
            exit()


def train(loader_train, model, criterion, optimizer):
    loss_sum = 0
    model.train()
    num_correct = 0
    num_samples = 0
    result_table = np.zeros((3,3))
    for t, (X, y) in enumerate(loader_train):
        X_var = Variable(X.cuda(args.gpu))
        y_var = Variable(y.cuda(args.gpu)).long()
        scores = model(X_var)
        loss = criterion(scores, y_var)
        loss_sum += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        for i in range(preds.shape[0]):
            result_table[preds[i], y[i]] += 1
    print(result_table)
    acc = float(num_correct) / num_samples
    print('train acc : (%.2f)' % (100 * acc))

    return acc

def check_accuracy(model, loader_test):
    num_correct = 0
    num_samples = 0
    model.eval()
    result_table = np.zeros((3,3))

    for t, (X, y) in enumerate(loader_test):
        X_var = Variable(X.cuda(args.gpu))
        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        for i in range(preds.shape[0]):
            result_table[preds[i], y[i]] += 1
    print(result_table)
    acc = float(num_correct) / num_samples
    print('test acc : (%.2f)' % (100 * acc))
    return acc

if __name__ == '__main__':
    main()
