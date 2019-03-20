import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os.path


def detect(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image_resized_data = cv2.resize(image,(224,224))
    
    return image_resized_data
        #cv2.imshow("AnimeFaceDetect", image_face_data)
        #cv2.waitKey(0)


def estimate(image_data,model_ft,use_gpu,classes):
    model_ft.eval()
    if use_gpu:
        images = Variable(image_data.cuda(), volatile=True)
    else:
        images = Variable(image_data,volatile=True)

    #推論
    outputs = model_ft(images)
    #parsent = F.softmax(outputs,dim=1)[0].detach().cpu().numpy()
    parsent = F.softmax(outputs,dim=1)[0]
    print(parsent)
    #print("hasi : {}% , hisi : {}% , hune : {}% , kaeru : {}% , kawa : {}% , tanbo : {}% , tudumi : {}%".format(outputs.data[0][0]*100.0/total_data, outputs.data[0][1]*100.0/total_data,outputs.data[0][2]*100.0/total_data,outputs.data[0][3]*100.0/total_data,outputs.data[0][4]*100.0/total_data,outputs.data[0][5]*100.0/total_data,outputs.data[0][6]*100.0/total_data))
    print("hasi : {:.2f}% , hisi : {:.2f}% , hune : {:.2f}% , kaeru : {:.2f}% , kawa : {:.2f}% , tanbo : {:.2f}% , tudumi : {:.2f}%".format(parsent[0]*100.0, parsent[1]*100.0,parsent[2]*100.0,parsent[3]*100.0,parsent[4]*100.0,parsent[5]*100.0,parsent[6]*100.0))

    _, predicted = torch.max(outputs.data,1)
    print(classes[predicted.item()])


def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    #print(images)
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    plt.show()

def main():
    data_transform = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(224),#訓練データを224にリサイズする
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-180,180)),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ]),
        'val': transforms.Compose([
        #transforms.RandomResizedCrop(224),#推定ではもとデータを224にするのでいらない
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])
    }
    
    
    classes = ('hasi','hisi','hune','kaeru','kawa','tanbo','tudumi')
    
    #引数に与えたファイルを読み込み推定する
    image_data = detect(sys.argv[1])
    image_data = Image.fromarray(image_data[:,:,::-1])
    x_image = data_transform['val'](image_data)
    #data_set = torch.utils.data.TensorDataset(x_image)
    #data_loader = torch.utils.data.DataLoader(data_set)
    #print(next(iter(data_loader)))
    images = torchvision.utils.make_grid(x_image)
    imshow(images)
    x_image = x_image.unsqueeze(0)
    
    #モデルの作成&重みの読み込み------------
    #model_ft = models.resnet152(pretrained=True) #このままだと1000クラス分類なので512->1000
    model_ft = models.resnet18(pretrained=True) #このままだと1000クラス分類なので512->1000
    #for param in model_ft.parameters():
    #    param.requires_grad = False
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,7) #これにより512->7の層に変わった
    
    #重みの読み込み
    param = torch.load('weight_1.pth')
    model_ft.load_state_dict(param)
    
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    
    if use_gpu:
        model_ft.cuda()
    #---------------------------------------
    estimate(x_image,model_ft,use_gpu,classes)

if __name__ == "__main__":
    main()
