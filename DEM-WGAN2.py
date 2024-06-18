import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

data = pd.read_csv(
    r'D:\PycharmProjects\不平衡数据 数据型分类\vehi\vehicle0.dat', header=None)
print(data)
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
labeldata = np.array(le.transform(y_label)).reshape(-1, 1)
columnstestdata = data.shape[1]-1
testdata = pd.concat([data.iloc[:, 0:columnstestdata], pd.DataFrame(labeldata)], axis=1)
testdata.columns = [i for i in range(0, columnstestdata + 1)]

'-----input the majority class data -----'
majdata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==0,0:columnstestdata-1])
real_data=torch.Tensor(majdata.values)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

'---创建生成器与判别器-----'
'判别器'
class Discriminator1(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Discriminator1, self).__init__()
        self.disc=nn.Sequential(nn.Linear(input_size,hidden_size),
                                nn.Linear(hidden_size, 512),
                                nn.LeakyReLU(0.1),
                                nn.Linear(512, hidden_size),
                                nn.LeakyReLU(0.1),
                                nn.Linear(hidden_size,output_size),
                                )

    def forward(self,disc_data):
        dic_output=self.disc(disc_data)
        return dic_output
'生成器'
class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        '''input_size 是指输入到生成器数据的维度，可以自定义，
        output_size是指输出到判别器的维度必须和源数据的维度相同，因为此时判别器需要判断是真数据还是假数据'''
        super(Generator, self).__init__()
        self.gen=nn.Sequential(nn.Linear(input_size,hidden_size),
                               nn.Linear(hidden_size,128),
                               # nn.Linear(128, 256),
                               nn.Linear(128, hidden_size),
                               # nn.Linear(256, hidden_size),
                               nn.LeakyReLU(0.1),
                               nn.Linear(hidden_size,output_size),
                               )

    def forward(self,gen_data):
        gen_data_output=self.gen(gen_data)
        return gen_data_output


#测试生成器与判别器
'---规定参数----'
G_input_size=64
G_hidden_size=128
G_output_size=columnstestdata
gen=Generator(input_size=G_input_size,hidden_size=G_hidden_size,output_size=G_output_size).to(device)

'-------'
D_input_size=columnstestdata
D_hidden_size=128
D_output_size=1
disc=Discriminator1(input_size=D_input_size,hidden_size=D_hidden_size,output_size=D_output_size).to(device)

'---定义优化算法---'
optim_gen=optim.RMSprop(gen.parameters(),lr=0.001)
optim_disc=optim.RMSprop(disc.parameters(),lr=0.00001)

'---定义损失函数---'
# criterion=nn.BCELoss()   #采样上述损失函数

'''----参数迭代----'''
epochs=500
batch=majdata.shape[0]
lossG=[]
lossD=[]
loss_D1=[]
G_mean=[]
G_std=[]
loss_Real=[]
loss_Fake=[]
for epoch in range(epochs):
    '''对数据进行切分，每一次得到batch个数据'''
    '''训练分类器'''
    for i in range(10):
        #train on generator
        # stat=i*batch
        # end=stat+batch
        # '''判别器的损失'''
        # x_real_data=real_data[stat:end]
        optim_disc.zero_grad()
        disc_real_data=disc(real_data)

        #train on fake
        noise=torch.randn((batch,G_input_size))
        gen_data1=gen(noise)
        gen_data2=disc(gen_data1.detach())
        loss_D=-torch.mean(disc_real_data)+torch.mean(gen_data2)
        loss_D1.append(loss_D.detach().numpy())
        loss_D.backward()
        optim_disc.step()

        for p in disc.parameters():
            p.data.clamp_(-0.01, 0.01)

    lossD.append(loss_D1[-1])
    '''生成器的损失'''
    ##生成器的反向传播
    optim_gen.zero_grad()
    noise = torch.randn((batch, G_input_size))
    gen_data4 = gen(noise)
    gen_data3=disc(gen_data4)
    loss_G=-torch.mean(gen_data3)
    lossG.append(loss_G.detach().numpy())
    loss_G.backward()
    optim_gen.step()
    with torch.no_grad():
        G_mean.append(np.mean(gen_data4.data.numpy(),axis=0))
        G_std.append(np.cov(gen_data4.data.numpy(),rowvar=False))

    if epoch % 10==0:
        print("mean:{},std:{},Epoch: {}, loss_D:{} ,loss_G:{}"
              .format(G_mean[-1],G_std[-1],epoch,lossD[-1],lossG[-1]))



print(disc.state_dict().keys())  # 输出模型参数名称
#保存模型参数到路径"./data/model_parameter.pkl"
torch.save(disc.state_dict(), "D:\PycharmProjects\sci4\程序\model_parameter1.pth")
new_model = Discriminator1(input_size=D_input_size,hidden_size=D_hidden_size,output_size=D_output_size)  # 调用模型Model
new_model.load_state_dict(torch.load("D:\PycharmProjects\sci4\程序\model_parameter1.pth"))  # 加载模型参数
numpy1=-1*new_model(real_data).detach().numpy()
df1=pd.DataFrame(numpy1)
df1.to_csv(r'D:\PycharmProjects\sci4\数据\maj.csv',index=False)


'''loss函数画图'''
plt.plot(lossG,c='green',label='loss G')
plt.title('Loss Function')
plt.plot(lossD,c='red',label='loss D')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(linestyle='-.')  #设置网格
plt.xticks(range(0, 510, 100))
plt.show()

