
import pandas as pd 
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import time
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from MyDataset import *
from sklearn.model_selection import KFold
################################################################################################################################################
# import wandb
# wandb.init(project="scene-classification")
# config = wandb.config
# config.dropout = 0.01

train_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       # transforms.Normalize([0.485, 0.456, 0.406],
                                       #                      [0.229, 0.224, 0.225])
                                       transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5))])

test_transforms = transforms.Compose([
                                        transforms.Resize((224,224)),
                                      # transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5))])
                                      # transforms.Normalize([0.485, 0.456, 0.406],
                                      #                      [0.229, 0.224, 0.225])])


# generate_txt(data_dir,label_file,test_file)
train_data = MyDataset(txt='./train1.txt',type = "train", transform=train_transforms)
test_data  = MyDataset(txt='./test1.txt', type = "test", transform=test_transforms)

#test集可以直接dataloader
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


batch_size = 32
print('train_data:',len(train_data))
print('test_data:',len(test_data))
# validation_split = 0.2
# shuffle_dataset = True
# random_seed= 42
# dataset_size = len(train_data)
# print('-'*100)
# print('dataset_size:',dataset_size)
train_dataset,test_dataset = train_test_split(train_data, test_size=0.2, random_state=42)
print('train_dataset:',len(train_dataset))
print('val_dataset:',len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)   
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)   

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)  


# print(type(train_loader))
# print(type(test_loader))



########################################################################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)


model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


def train(model,criterion):
    # wandb.watch(model)
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#SGD（螺旋）随机梯度下降法，
    
    # wandb.watch(model)
    learning_rate = 0.001
    
    decay=0.1
    num_epochs = 20
    avg_loss = 0
    cnt = 0
    total=0
    correct=0
    for epoch in range(0, num_epochs):
        #print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        start = time.time()
        for i,data in enumerate(train_loader,0):
            
            images,labels = data
            images,labels = images.to(device),labels.to(device)  
            
            optimizer.zero_grad()
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            correct+=(predicted == labels).sum().item()
            
            loss = criterion(outputs,labels)
            
            avg_loss += loss.data
            cnt += 1
            
            
            loss.backward()
            optimizer.step()
            
            current_loss+=loss.item()

        # train_accuracy = 100 * correct / total
            
    
        print('[Epoch:{}] Loss:{:.5f}'.format(epoch+1,loss.item()))
        # print('Accuracy of the network on the train images: %f %%' % (100 * correct / total))
            
        # wandb.log({"loss": loss.data})
        # wandb.log({"Avg loss": avg_loss/cnt})
        # if epoch%10==0:
        #     print('learning rate:',learning_rate)
        
        # if epoch < 10 :
        #     learning_rate=0.0001
        # elif epoch < 60:
        #     learning_rate=0.001
        # elif epoch%10==0:
            
        #     learning_rate*=decay
        # elif epoch<120:
        #       learning_rate*=decay
        # wandb.log({"loss": loss.data})
        # wandb.log({"Avg loss": avg_loss/cnt})
        print('-----------------------------------------------')
        # print('Saving trained model...')
        # torch.save(model.state_dict(), './weight/vgg16_cls_scene_1')
        # torch.save(model.state_dict(), './weight/alexnet_cls_scene_')
        with torch.no_grad():
            model.eval()
            total=0
            correct=0
            for i,data in enumerate(val_loader,0):
                images,labels = data
                images,labels = images.to(device),labels.to(device)  
                # Generate outputs
                outputs = model(images)       
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # Print accuracy
            # test_accuracy = 100 * correct / total
            print('Accuracy of the network on the test  images: %f %%' % (100 * correct / total))
        
        end = time.time()
        torch.cuda.synchronize()
        print('Use time:',end-start)
        # wandb.log({"Accuracy": 100.0 * correct / total})
    print('-----------------------------------------------')



def test(model,criterion):
    # PATH = './weight/vgg16_cls_scene_'
    # PATH = './weight/alexnet_cls_scene_'
    
    model.load_state_dict(torch.load(PATH))
    
    with torch.no_grad():
            model.eval()
            total=0
            correct=0
            for i,data in enumerate(test_loader,0):
                images,labels = data
                images,labels = images.to(device),labels.to(device)  
                # Generate outputs
                outputs = model(images)       
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # Print accuracy
            # test_accuracy = 100 * correct / total
            print('Accuracy of the network on the test  images: %f %%' % (100 * correct / total))

if __name__ == '__main__':
    print("Start Training!!")
    train(model,criterion)
    # test(model,criterion)

