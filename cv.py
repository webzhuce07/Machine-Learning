import glob, json
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path, self.img_label, self.transform = img_path, img_label, transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB') # 读取数据
        img = self.transform(img) # 做相应变换
        if self.img_label:
            lbl = np.array(self.img_label[index], dtype=np.int) # 制作标签
            lbl = list(lbl)  + (5 - len(lbl)) * [10] # 标签长度少于五的用10来填充
            return img, torch.from_numpy(np.array(lbl[:5]))
        else:
            return img

    def __len__(self):
        return len(self.img_path)

# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        self.cnn = models.resnet50(pretrained=True) # 加载resnet50
        self.cnn.avgpool = nn.AdaptiveAvgPool2d(1) # 将平均池化改为自适应平均池化
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1]) # 去除最后的线性层
        self.fc1,self.fc2,self.fc3 = nn.Linear(2048, 11), nn.Linear(2048, 11), nn.Linear(2048, 11)
        self.fc4,self.fc5 = nn.Linear(2048, 11), nn.Linear(2048, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1,c2,c3 = self.fc1(feat), self.fc2(feat), self.fc3(feat)
        c4,c5 = self.fc4(feat), self.fc5(feat)
        return c1, c2, c3, c4, c5

def train(train_loader, model, criterion, optimizer):
    model.train() # 切换模型为训练模式
    train_loss = []
    for input, target in tqdm(train_loader): # 取出数据与对应标签
        if use_cuda: # 如果是gpu版本
            input, target = input.cuda(), target.cuda()
        target = target.long()
        c0, c1, c2, c3, c4 = model(input) # 得到预测值
        loss = criterion(c0, target[:, 0]) + criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4]) # 计算loss
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 参数更新
        train_loss.append(loss.item())
    return np.mean(train_loss)

def predict(test_loader, model):
    model.eval() # 切换模型为预测模型
    test_pred = []
    with torch.no_grad(): # 不记录模型梯度信息
        for input in tqdm(test_loader):
            if use_cuda: input = input.cuda()
            c0, c1, c2, c3, c4 = model(input)
            if use_cuda:
                output = np.concatenate([
                    c0.data.cpu().numpy(), c1.data.cpu().numpy(), c2.data.cpu().numpy(), # 将结果水平合并，即第一个字符索引为第一列到第十一列，
                    c3.data.cpu().numpy(), c4.data.cpu().numpy()], axis=1)               # 第二个字符为第十二列到第二十二列，依次往下
            else:
                output = np.concatenate([
                        c0.data.numpy(), c1.data.numpy(), c2.data.numpy(),
                        c3.data.numpy(), c4.data.numpy()], axis=1)
            test_pred.append(output)
        test_pred = np.vstack(test_pred) # 将每个batch的结果垂直堆起来
    return test_pred

train_path, test_path = glob.glob('../input/train/*.png'), glob.glob('../input/test_a/*.png') # 读取训练数据和测试数据
train_path.sort(); test_path.sort()

train_json = json.load(open('../input/train.json'))  #读取训练集标注文件
train_label = [train_json[x]['label'] for x in train_json] # 拿出训练集的标签

trans_fun = transforms.Compose([
                transforms.Resize((64, 128)), # 将图片裁剪为64*128
                transforms.ToTensor(),  #转为Tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
])

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label, trans_fun),
    batch_size=40, shuffle=True) # 批量大小40，打乱顺序

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, [], trans_fun),
    batch_size=40, shuffle=False)

model = SVHN_Model1()
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), 0.001) # Adam优化器

use_cuda = torch.cuda.is_available()
if use_cuda: model = model.cuda()

for epoch in range(10):
    train_loss = train(train_loader, model, criterion, optimizer) # 训练
    print(epoch, train_loss)
test_predict_label = predict(test_loader, model)
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1), test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1), test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x!=10]))) # 取出预字符不为10的字符且顺序排列

df_submit = pd.read_csv('../input/sample_submit_A.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('submit.csv', index=None) # 保存结果文件