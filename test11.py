import torch
from minist11 import *
import glob
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torchvision

# from skimage import io, transform

# if __name__ == '__main__':
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./MNIST.pth')  # 加载模型
model = model.to(device)
model.eval()  # 把模型转为test模式

# 循环读取文件夹内的jpg图片并输出结果
for jpgfile in glob.glob(r'./*.jpg'):
    print(jpgfile)  # 打印图片名称，以与结果进行对照
    img = cv2.imread(jpgfile)  # 读取要预测的图片，读入的格式为BGR
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
    img = torch.from_numpy(img)
    img = img.to(device)
    output = model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())

# pytorch环境中
model_pth = 'MNIST.pth'  # 模型的参数文件
mobile_pt = 'MNIST.pt'  # 将模型保存为Android可以调用的文件

model = torch.load(model_pth)
model.eval()  # 模型设为评估模式
device = torch.device('cpu')
model.to(device)
# 1张3通道224*224的图片
input_tensor = torch.rand(1, 1, 28, 28)  # 设定输入数据格式

mobile = torch.jit.trace(model, input_tensor)  # 模型转化
mobile.save(mobile_pt)  # 保存文件
