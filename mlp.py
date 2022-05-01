import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import random
import focal_loss

# Training settings
batch_size = 10000
# read matrix
train_Matrix = []
train_label = []
split_Matrix = []
split_label = []

def readDATASET():
    global train_Matrix,train_label,split_Matrix,split_label
    ### huafen ce shi ji xun lian ji

    print("read matrix train")
    #Ftrain = open("bert_text/save_encoding_sen_review_train.txt","r")
    Ftrain = open("vector_feature_train.txt", "r")
    Ftrain1 = open("save_encoding_sen_Amenity_train.txt", "r")


    ii = 0
    feature=Ftrain.readlines()
    Amenity=Ftrain1.readlines()
    for x, y in zip(feature,Amenity):
        if x != "\n":
            t = x.replace(" \n","").split(" ")+y.replace(" \n","").split(" ")
            #print(len(t))
            for i in range(0,len(t)):
                t[i] = float(t[i])
            train_Matrix.append(t)
            ii += 1
            if ii % 1000 == 0:
                print(ii)

    print("read label train")
    Ftrainlabel = open("train_label.txt", "r")
    T0, T1, T2, T3, T4, T5 = [],[],[],[],[],[]
    ii = 0
    for l in Ftrainlabel.readlines():
        if l != "\n":
            l = l.replace("\n","")
            train_label_temp = 0.0
            if l == "0":
                train_label_temp = 0.0
                T0.append(ii)
            elif l == "1":
                train_label_temp = 1.0
                T1.append(ii)
            elif l == "2":
                train_label_temp = 2.0
                T2.append(ii)
            elif l == "3":
                train_label_temp = 3.0
                T3.append(ii)
            elif l == "4":
                train_label_temp = 4.0
                T4.append(ii)
            elif l == "5":
                train_label_temp = 5.0
                T5.append(ii)
            else:
                print("error!")
            train_label.append(train_label_temp)
            ii += 1
    #print(train_label)
    # print(len(train_label))
    print(len(T0), len(T1), len(T2),len(T3),len(T4),len(T5))

    #### split ######
    random.seed(10)
    splitT0 = random.sample(T0, int(len(T0) * 0.2))
    splitT1 = random.sample(T1, int(len(T1) * 0.2))
    splitT2 = random.sample(T2, int(len(T2) * 0.2))
    splitT3 = random.sample(T3, int(len(T3) * 0.2))
    splitT4 = random.sample(T4, int(len(T4) * 0.2))
    splitT5 = random.sample(T5, int(len(T5) * 0.2))


    splitx = splitT0 + splitT1 + splitT2+splitT3 + splitT4+splitT5
    #print(splitx)
    print(len(splitT0) , len(splitT1), len(splitT2),len(splitT3),len(splitT4),len(splitT5))

    split_Matrix = [train_Matrix[i] for i in splitx]
    split_label = [train_label[i] for i in splitx]

    train_Matrix = [train_Matrix[i] for i in range(0,len(train_Matrix)) if i not in splitx]
    train_label = [train_label[i] for i in range(0,len(train_label)) if i not in splitx]
    # chong cai yang

    print(len(train_Matrix), len(split_Matrix))


# MNIST Dataset
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(960, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 6)

    def forward(self, din):
        dout = torch.nn.functional.relu(self.fc1(din))
        dout = torch.nn.functional.relu(self.fc2(dout))
        dout = torch.nn.functional.relu(self.fc3(dout))
        dout = self.fc4(dout)
        return dout

print("Read DataSet")
readDATASET()
x_train, y_train = Variable(torch.Tensor(train_Matrix)),Variable(torch.Tensor(train_label).to(dtype=torch.long))
x_test , y_test = Variable(torch.Tensor(split_Matrix)),Variable(torch.Tensor(split_label).to(dtype=torch.long))
print(x_train,y_train)
print(x_train.shape,y_train.shape)
print(x_test,y_test)
print(x_test.shape,y_test.shape)

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size=batch_size)
#
model = MLP()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1, momentum = 0.99)

w = torch.tensor([1,1,2.8,2.2,3.6,12],dtype=torch.float)
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.CrossEntropyLoss(weight= w)
# loss_fn = focal_loss.FocalLoss(class_num = 3)

def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np),  np.argmax(pred, axis=1)

def static(a):
    one, tow, three = 0, 0 ,0
    for x in a:
        if x == 0:
            one += 1
        elif x == 1:
            tow += 1
        else:
            three += 1
    print(one, tow, three)

global_loss = []
global_F1 = []

def train(epoch):
    print("epoch " + str(epoch))
    # 每次输入barch_idx个数据
    print("start training!")
    avg_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        # loss
        loss = loss_fn(output, target)
        loss.backward()
        # update
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    outputtotal = model(x_test)
    acc, ypre = AccuarcyCompute(outputtotal, y_test)
    print("acc:", acc)
    static(list(ypre))
    static(list(y_test))
    #print((ypre), target)
    oriF1 = f1_score(y_test, ypre, average="macro")
    oriF2 = f1_score(y_test, ypre, average="micro")
    print("sklearn-f1-macro:", oriF1, " micro:",oriF2)
    print(avg_loss)
    global_loss.append(avg_loss)
    global_F1.append(oriF1)

if __name__ == '__main__':
    for epoch in range(0,30):
        train(epoch)

    print(global_loss)
    print(global_F1)
