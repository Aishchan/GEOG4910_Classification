import os
import time
from importlib.resources import path
from cv2 import imshow
from nbformat import write

import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import TM_Dataset
from metrics import acc_metric
from models.FCN import FCN8s, VGGNet

# load config file
path_to_yaml = './configs/TM_Dataset_FCN.yaml'

try:
    with open (path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file!')

logs_folder = os.path.join(config['experiments']['save_directory'],config['experiments']['experiments_name'])

if os.path.exists(logs_folder) == False:
        os.makedirs(logs_folder)
else:
    raise FileNotFoundError('{} is already exist!'.format(logs_folder))


LABELS = config['dataset_info']['class_name']
N_CLASSES = config['dataset_info']['class_number'] 
IN_CHANNELS = config['dataset_info']['input_channel_number']

DATA_DIR = config['dataset_info']['dataset_path']

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')


BATCH_SIZE = config['hyperparamters']['batch_size']

if config['dataset_info']['dataset_name'] == 'TM_Dataset':
    train_dataset = TM_Dataset(images_dir=x_train_dir, masks_dir=y_train_dir)
    valid_dataset = TM_Dataset(images_dir=x_valid_dir, masks_dir=y_valid_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if config['model_info']['model_name'] == 'FCN':
    encoder = VGGNet('vgg16',batch_norm=True,IN_CHANNELS=IN_CHANNELS)
    model = FCN8s(encoder, N_CLASSES)

else:
    raise Exception('No model found!')
    
if config['train_info']['loss_function'] == 'Cross Entropy Loss':
    criterion = nn.CrossEntropyLoss()
else:
    raise Exception('Loss function is not found!')
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
metric = acc_metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train_loop(model, optimizer, criterion, metric,train_loader, epoch):
    
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for batch, (inputs, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        acc = metric(outputs, labels)
              
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc.item() * inputs.size(0)

        if batch % 10 == 0:
            print('Training epoch {} step {} -- Loss: {}  Acc: {}'.format(epoch+1,batch, loss, acc))

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch__train_acc = running_acc / len(train_loader.dataset)

    return epoch_train_loss, epoch__train_acc

@torch.no_grad()
def valid_loop(model, criterion, metric, valid_loader, epoch):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for batch, (inputs, labels) in enumerate(valid_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        acc = metric(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_acc += acc.item() * inputs.size(0)
        
        if batch % 10 == 0:
            print('Validating Epoch {} step {} -- Loss: {}  Acc: {}'.format(epoch+1, batch, loss, acc))

    epoch_valid_loss = running_loss / len(valid_loader.dataset)
    epoch_valid_acc = running_acc / len(valid_loader.dataset)

    return epoch_valid_loss, epoch_valid_acc, inputs, outputs, labels

def train(model, criterion, optimizer, metric, train_loader, valid_loader,EPOCHS):
    print('Start training')
    
    start_time = time.time()

    best_acc = 0.0

    train_loss_list, valid_loss_list = [], []
    train_acc_list, valid_acc_list = [], []

    model.to(device)

    writer = SummaryWriter(log_dir=os.path.join(logs_folder,'tensorborad_stat'))

    for epoch in range(EPOCHS):

        print('-' * 30 + '\n' + 'Start epoch {}/{} training'.format(epoch + 1, EPOCHS),'\n' + '-' * 30)

        train_loss, train_acc = train_loop(model, optimizer, criterion, metric, train_loader,epoch)

        writer.add_scalars('loss', {'train': train_loss}, epoch + 1)
        writer.add_scalars('acc', {'train': train_acc}, epoch + 1)

        print('-' * 30 + '\n' + 'Start epoch {}/{} validation'.format(epoch + 1, EPOCHS), '\n' + '-' * 30)

        valid_loss, valid_acc, valid_inputs, valid_outputs, valid_labels= valid_loop(model, criterion, metric, valid_loader, epoch)
        
        valid_pred = torch.unsqueeze(torch.argmax(valid_outputs, dim=1),dim=1)
        valid_labels = torch.unsqueeze(valid_labels, dim=1).to(torch.int64)
        

        print(valid_pred.dtype)
        print(valid_labels.dtype)


        valid_inputs_grid = torchvision.utils.make_grid(valid_inputs[ :, 0:3,: , :])
        valid_preds_grid = torchvision.utils.make_grid(valid_pred)
        valid_labels_grid = torchvision.utils.make_grid(valid_labels)

        writer.add_scalars('loss', {'valid': valid_loss}, epoch + 1)
        writer.add_scalars('acc', {'valid': valid_acc}, epoch + 1)
        
        writer.add_image('images', valid_inputs_grid, epoch + 1)
        writer.add_image('predictions', valid_preds_grid, epoch + 1)
        writer.add_image('labels', valid_labels_grid, epoch + 1)

        print('-' * 30 + '\n' + 'Epoch {}/{} train loss: {} train acc: {}'.format(epoch + 1, EPOCHS, train_loss, train_acc))
        print('Epoch {}/{} valid loss: {} valid Acc: {}'.format(epoch + 1, EPOCHS, valid_loss, valid_acc))
        # exp_lr_scheduler.step()
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model, os.path.join(logs_folder, 'best_model_epoch_{}.pth'.format(epoch+1)))
            print('Saving best model')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

    plt.figure(figsize=(10,5))
    plt.plot(train_loss_list, label='train loss')
    plt.plot(valid_loss_list, label='valid loss')
    plt.legend()
    plt.savefig(os.path.join(logs_folder,'loss.png'))

    plt.figure(figsize=(10,5))
    plt.plot(train_acc_list, label='train acc')
    plt.plot(valid_acc_list, label='valid acc')
    plt.legend()
    plt.savefig(os.path.join(logs_folder,'acc.png'))

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    writer.close()


    return model, train_loss_list, valid_loss_list


if __name__ == '__main__':
    train(model, criterion, optimizer, metric, train_loader, valid_loader, EPOCHS=5)
