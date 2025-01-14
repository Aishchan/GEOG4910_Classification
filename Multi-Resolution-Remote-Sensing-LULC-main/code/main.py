import os
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.pyplot import step
from torch.optim import lr_scheduler

from datasets import TM_Dataset
from losses import CrossEntropy2d
from metrics import acc_metric
from models.FCN import FCN8s, VGGNet
from utils import reverse_one_hot

# define parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
IN_CHANNELS = 7 # Number of input channels (e.g. RGB)
BATCH_SIZE = 2 # Number of samples in a mini-batch

LABELS = ["class1", "class2", "class3", "class4"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory



# define data path

DATA_DIR = 'D://projects_jyshao//mrrs//data//tm_dataset'
# DATA_DIR = '..//data//tm_dataset'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')


# load data
train_dataset = TM_Dataset(images_dir=x_train_dir, masks_dir=y_train_dir)
valid_dataset = TM_Dataset(images_dir=x_valid_dir, masks_dir=y_valid_dir)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)


os.environ["CUDA_VISIBLE_DEVICES"]='0'
encoder = VGGNet('vgg16',batch_norm=True,IN_CHANNELS=IN_CHANNELS)
model = FCN8s(encoder, N_CLASSES)

# define hyperparameters

EPOCHS = 5
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
# loss_func = CrossEntropy2d

optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



def train_loop(model, optimzer, criterion, metric,train_loader, epoch):
    
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for batch, (inputs, labels) in enumerate(train_loader):
        
        optimzer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        acc = metric(outputs, labels)
              
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0)

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
        running_acc += acc * inputs.size(0)
        
        if batch % 10 == 0:
            print('Validating Epoch {} step {} -- Loss: {}  Acc: {}'.format(epoch+1, batch, loss, acc))

    epoch_valid_loss = running_loss / len(valid_loader.dataset)
    epoch_valid_acc = running_acc / len(valid_loader.dataset)

    return epoch_valid_loss, epoch_valid_acc

def trainer(model, criterion, optimizer, metric, train_loader, valid_loader,EPOCHS):
    print('Start training')
    
    start_time = time.time()
    model.to(device)
    for epoch in range(EPOCHS):

        print('-' * 30 + '\n' + 'Start epoch {}/{} training'.format(epoch + 1, EPOCHS),'\n' + '-' * 30)

        train_loss, train_acc = train_loop(model, optimizer, criterion, metric, train_loader,epoch)
        
        print('-' * 30 + '\n' + 'Start epoch {}/{} validation'.format(epoch + 1, EPOCHS), '\n' + '-' * 30)

        valid_loss, valid_acc = valid_loop(model, criterion, metric, valid_loader, epoch)
        
        print('-' * 30 + '\n' + 'Epoch {}/{} train loss: {} train acc: {}'.format(epoch + 1, EPOCHS, train_loss, train_acc))
        print('Epoch {}/{} valid loss: {} valid Acc: {}'.format(epoch + 1, EPOCHS, valid_loss, valid_acc))
        # exp_lr_scheduler.step()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

# def train_model(model, criterion, acc_metric, optimizer, train_loader, valid_loader, scheduler, epochs=EPOCHS):
def train_model(model, criterion, acc_metric, optimizer, train_loader, valid_loader, epochs=EPOCHS):

    since = time.time()
    model.to(device)
    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:

            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            running_loss = 0.0
            running_acc = 0.0
            
            step = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                
                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_acc += acc_metric(outputs, labels) * inputs.size(0)

                if step % 10 == 0:
                    # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss / (step * BATCH_SIZE), running_acc / (step * BATCH_SIZE)))
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc_metric(outputs, labels), torch.cuda.memory_allocated()/1024/1024))

                step += 1

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def train(model, criterion, optimizer, train_loader, valid_loader, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_loader
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_loader

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y.long())

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = criterion(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    


# def test(model_path, img_folder_path, mask_folder_path):
    
#     return test_acc

def test_big(model_path, img_path, mask_path):
    
    # load model
    best_model = torch.load(model_path, map_location=device)
    
    # read img and mask and preprocessing
    rsimage = rasterio.open(img_path) # read img by rasterio
    image = rsimage.read([1,2,3,4,5,6,7]) # load bands
    H, W = image.shape
    image = 1/255 * np.asarray(image, dtype='float32') # normalize images
    image = image[:,0:H,0:W] # clip img to multiples of 32
    # image_vis = np.transpose(image,(1,2,0))
    rslabel = rasterio.open(mask_path) # open tif by rasterio
    label = rslabel.read(1) # load label image


    # convert label from 0-3(4 classes)
    label[label==1] = 0
    label[label==2] = 1
    label[label==3] = 2
    label[label==4] = 3
    label = np.expand_dims(label,0)
    label = label[:,0:H,0:W] # clip label to the same dimention of input img
    rsimage.close()
    rslabel.close()
    
    # prediction
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0) # input tensor
    pred_mask = best_model(x_tensor) # get predicted map from trained model
    pred_mask = pred_mask.detach().squeeze().cpu().numpy() # move pred img to cpu
    pred_mask = np.transpose(pred_mask,(1,2,0))
    predicted = reverse_one_hot(pred_mask)
    # gt_mask = np.transpose(label,(1,2,0))
    
    # accuracy assessment
    pred_list = predicted.flatten()
    gt_list = label.flatten()
    # metrics(pred_list,gt_list)


# def inference(model_path, img_folder_path):

#     return inf_result


# def inference_big(model_path, img_folder_path):

#     return inf_result

if __name__ == '__main__':

    # train_model(model, criterion, acc_metric=acc_metric, optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader, epochs=10)

    # train(model, criterion, optimizer, train_loader, valid_loader, acc_fn=acc_metric, epochs=10)

    trainer(model, criterion, optimizer, metric = acc_metric, train_loader=train_loader, valid_loader=valid_loader, EPOCHS=2)
