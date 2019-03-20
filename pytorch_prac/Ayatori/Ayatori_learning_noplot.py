import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    #plt.show()
    plt.savefig("image_learning_test.png")

#訓練データの学習
def train(train_loader):
    scheduler.step()
    model_ft.train()
    running_loss = 0
    for batch_idx, (images,labels) in enumerate(train_loader):
        if use_gpu:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model_ft(images)

        loss = criterion(outputs,labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)

    return train_loss

#テストデータに対する精度を見る
def valid(test_loader):
    model_ft.eval()
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images,labels) in enumerate(test_loader):
        if use_gpu:
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
        else:
            images = Variable(images,volatile=True)
            labels = Variable(labels,volatile=True)

        outputs = model_ft(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels.data).sum()
        total += labels.size()[0]

    correct = float(correct)
    total   = float(total)
    val_loss = running_loss / len(test_loader)
    val_acc  = correct / total
    #print(val_acc)

    return(val_loss,val_acc)

    
data_transform = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-180,180)),
    #transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ]),
    'val': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
}

# train data読み込み
#ここでバッチサイズやイテレーション数などを設定する
hymenoptera_dataset = datasets.ImageFolder(root='waza_dataset/train',
                                           transform=data_transform['train'])
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

# test data読み込み
#hymenoptera_testset = datasets.ImageFolder(root='hymenoptera_data/val',
#                               transform=data_transform['val'])
#dataset_testloader = torch.utils.data.DataLoader(hymenoptera_testset, batch_size=4,
#                                         shuffle=False, num_workers=4)

#todo フォルダ名からリスト化してソートする
classes = ('hasi','hisi','hune','kaeru','kawa','tanbo','tudumi')

images, classes_nam = next(iter(dataset_loader))
print(images.size(), classes_nam.size())  # torch.Size([4, 3, 224, 224]) torch.Size([4])
images = torchvision.utils.make_grid(images)
imshow(images, title=[classes[x] for x in classes_nam])

#modelの作成##################
model_ft = models.resnet18(pretrained=True) #このままだと1000クラス分類なので512->1000
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features,7) #これにより512->7の層に変わった
##############################

#model_ft = models.resnet18(pretrained=True) #このままだと1000クラス分類なので512->1000
use_gpu = torch.cuda.is_available()
num_epochs = 25
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001,momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if use_gpu:
    model_ft.cuda()


#学習開始
loss_list = []
val_loss_list = []
val_acc_list =[]

for epoch in range(num_epochs):
    loss = train(dataset_loader)
    #val_loss, val_acc = valid(dataset_testloader)

    #print('epoch : {}, loss : {:.4f}, val_loss : {:.4f}, val_acc : {:.4f}'.format(epoch,loss,val_loss, val_acc))
    print("epoch : {}, loss : {:.4f}".format(epoch,loss))
    #logging
    loss_list.append(loss)
    #val_loss_list.append(val_loss)
    #val_acc_list.append(val_acc)

#重みを保存する
torch.save(model_ft.state_dict(),'weight.pth')

plt.plot(range(num_epochs),loss_list)
#plt.plot(range(num_epochs),val_loss_list)
#plt.show()
plt.savefig("loss_list.png")
