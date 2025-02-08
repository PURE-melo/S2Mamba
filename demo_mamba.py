import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
from s2mamba import S2Mamba
from utils import *



parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU_Hi_LongKou'], default='Indian', help='dataset to use')
parser.add_argument('--flag', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--sess', default='mamba')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=100, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--epoches', type=int, default=400, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
setup_seed(args)
input_normalize, label, num_classes, TR, TE, color_matrix, color_matrix_pred = load_HSI(args)
height, width, band = input_normalize.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train, x_test = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
x_train=torch.from_numpy(x_train.transpose(0,3,1,2)).type(torch.FloatTensor) 
y_train=torch.from_numpy(y_train).type(torch.LongTensor) 
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test.transpose(0,3,1,2)).type(torch.FloatTensor) 
y_test=torch.from_numpy(y_test).type(torch.LongTensor) 
Label_test=Data.TensorDataset(x_test,y_test)
label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
# x_true=torch.from_numpy(x_true.transpose(0,3,1,2)).type(torch.FloatTensor)
# y_true=torch.from_numpy(y_true).type(torch.LongTensor)
# Label_true=Data.TensorDataset(x_true,y_true)
# label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
#-------------------------------------------------------------------------------
model = S2Mamba(
            in_chans=band, 
            patch=args.patches,
            num_classes=num_classes, 
            depths=[1], 
            dims=[64],
            drop_path_rate=args.dropout, 
            attn_drop_rate=args.dropout
        )
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
#-------------------------------------------------------------------------------
if args.flag == 'test':
    model = torch.load('./{}_{}.pt'.format(args.sess,args.dataset))  
    model.eval()
    start = time.time()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA, AA_mean, Kappa, AA = output_metric(tar_v, pre_v)
    print(OA, AA_mean, Kappa, AA)
    
elif args.flag == 'train':
    OA_ls = []
    for epoch in range(args.epoches): 
        start = time.time()
        scheduler.step()
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        print("train_acc: {:.4f} | loss: {:.4f}".format(train_acc, train_obj))
        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):   
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA, AA_mean, Kappa, AA = output_metric(tar_v, pre_v)
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
            OA_ls.append(OA.item())
    torch.save(model, '{}_{}.pt'.format(args.sess,args.dataset))
    print("**************************************************")
    print("Final result: OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))









