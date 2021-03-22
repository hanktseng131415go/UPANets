'''Train and Test UPANets with PyTorch.'''
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import sys
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
#total_epoch = 100

'''path setting'''
parser = argparse.ArgumentParser(description='Train and Test UPANets with PyTorch')
parser.add_argument('--pkg_path', default='./', type=str, help='package path')
parser.add_argument('--save_path', default='./results/', type=str, help='package path')

'''experiment setting'''
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--datasets', default='cifar_10', type=str, help='using dataset')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='total traing epochs')

'''model setting'''
parser.add_argument('--blocks', default=1, type=int, help='block number in UPANets')
parser.add_argument('--filters', default=16, type=int, help='filter number in UPANets')

args = parser.parse_args()

pkgpath = args.pkg_path
save_path = args.save_path

if os.path.isdir(save_path) == False:
    os.makedirs(save_path)

sys.path.append(pkgpath)

from models.upanets import UPANets
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
img_size = 32 # default image size for Cifar-10
if args.datasets == 'cifar_10' or args.datasets == 'cifar_100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    if args.datasets == 'cifar_10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar_10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar_10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        classes = 10
        img_size = 32
        
    if args.datasets == 'cifar_100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data/cifar_100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data/cifar_100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        classes = 100
        img_size = 32
        
elif args.datasets == 'tiny_imgnet':
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_dir = './data/tiny-imagenet-200/train'

    trainset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    test_dir = './data/tiny-imagenet-200/val'
    testset = torchvision.datasets.ImageFolder(
        test_dir, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    classes = 200
    img_size = 64
# Model
print('==> Building model..')
net =  UPANets(args.filters, classes, args.blocks, img_size)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
summary(net, (3, img_size, img_size))
#%%
# Training
def train(epoch):
    print('Epoch:{0}/{1}'.format(epoch, args.epochs))
    net.train()
    
    train_loss = 0 
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)        
        loss = criterion(outputs, targets)       
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'train_Loss: %.3f | train_Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc_list.append(100.*correct/total)
            
            progress_bar(batch_idx, len(testloader), 'test_Loss: %.3f | test_Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print()
            print('>>>best acc: {0}, mean: {1}, std: {2}'.format(best_acc, round(np.mean(acc_list), 2), round(np.std(acc_list), 2)))
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path+'/checkpoint'):
            os.mkdir(save_path+'/checkpoint')
        torch.save(state, save_path+'/checkpoint/ckpt.pth')
        best_acc = acc
        print('>>>best acc:', best_acc)
    
    return test_loss/(batch_idx+1), 100.*correct/total, best_acc

test_loss = 0
test_list = []
train_list = []
epoch_list = []
train_acc_list = []
test_acc_list = []
for epoch in range(start_epoch, start_epoch+args.epochs):
   
    epoch_list.append(epoch)
    
    train_loss, train_acc = train(epoch)
    train_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    test_loss, test_acc, best_acc = test(epoch)
    test_list.append(test_loss)
    test_acc_list.append(test_acc)
    
    epoch_line = 'epoch: {0}/ total epoch: {1} '.format(epoch, args.epochs) 
    best_acc_line = 'best_acc: {0} '.format(best_acc)
    accuracy_line = 'train_acc: {0} %, test_acc: {1} % '.format(train_acc, test_acc)
    loss_line = 'train_loss: {0},e test_loss: {1} '.format(train_loss, test_loss)
    
    if epoch % 1 == 0:
        plt.subplot(2, 1, 1)
        plt.plot(epoch_list, train_list, c = 'blue', label = 'train loss')
        plt.plot(epoch_list, test_list, c = 'red', label = 'test loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc=0)
        
        plt.subplot(2, 1, 2)
        plt.plot(epoch_list, train_acc_list, c = 'blue', label = 'train acc')
        plt.plot(epoch_list, test_acc_list, c = 'red', label = 'test acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(loc=0)
        
        plt.savefig(save_path+'/train_history.png')
#        plt.show()

    with open(save_path+'/logs.txt', 'a') as f:
        f.write(epoch_line + best_acc_line + accuracy_line + loss_line + '\n')
    scheduler.step()

