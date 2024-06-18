import warnings

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sci4.程序.wganmaj import Discriminator1
from smote import Smote
import numpy as np
warnings.filterwarnings("ignore")



data = pd.read_csv(
    r'D:\PycharmProjects\不平衡数据 数据型分类\vehi\vehicle0.dat', header=None)
print(data)
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
labeldata = np.array(le.transform(y_label)).reshape(-1, 1)
columnstestdata = data.shape[1]-1
data2 = pd.concat([data.iloc[:, 0:columnstestdata], pd.DataFrame(labeldata)], axis=1)
data2.columns = [i for i in range(0, columnstestdata + 1)]



'-----获取某一类数据-----'
mindata=pd.DataFrame(data2.loc[data2.iloc[:,-1]==1,0:columnstestdata-1])
majdata=pd.DataFrame(data2.loc[data2.iloc[:,-1]==0,0:columnstestdata-1])
X=mindata.values

#调用训练好的多数类模型
new_model1 = Discriminator1(input_size=columnstestdata,hidden_size=128,output_size=1)  # 调用模型Model
new_model1.load_state_dict(torch.load("D:\PycharmProjects\sci4\程序\model_parameter1.pth"))  # 加载模型参数


##
pro_min=pd.read_csv(r'D:\PycharmProjects\sci4\数据\min.csv',header=None,skiprows=1)
pro_maj=pd.read_csv(r'D:\PycharmProjects\sci4\数据\maj.csv',header=None,skiprows=1)
pro_min_mean=np.mean(pro_min.values,axis=0)
pro_min_std=np.std(pro_min.values,axis=0)
pro_maj_mean=np.round(np.mean(pro_maj.values,axis=0),5)
pro_maj_std=np.std(pro_maj.values,axis=0)

# s = Smote(sample=X, N=100, k=5)
# s.over_sampling()
# sy=pd.DataFrame(s.synthetic)
# print(sy)
result=[]
generate=majdata.shape[0]-mindata.shape[0]
# print(generate)
while len(result)<generate:
    s = Smote(sample=X, N=50, k=5)
    s.over_sampling()
    sy = pd.DataFrame(s.synthetic)
    for j in range(sy.shape[0]):
        num1 = np.round(-1 * new_model1(torch.Tensor(sy.values[j])).detach().numpy()[0],5)
        if num1 != pro_maj_mean:
            result.append(sy.values[j])
            if len(result)==generate:
                break
X_sample=pd.DataFrame(result)
X_sample.loc[:,columnstestdata]=[1]*generate
print(X_sample)
print('aa',generate)
df1=pd.concat([data2,X_sample],axis=0)
df1.to_csv(r'D:\PycharmProjects\sci4\数据\last.csv',index=False)
print(df1)

plt.scatter(majdata.values[:,0],majdata.values[:,1],alpha=0.5,label='majority class',
            facecolors='none',edgecolors='blue')
plt.scatter(mindata.values[:,0],mindata.values[:,1],alpha=0.5,c='green',label='minority class',
            marker='*')
plt.scatter(X_sample.iloc[:,0],X_sample.iloc[:,1],label='synthetic data', c='red', marker='x')
plt.title('DEM-WGAN')
plt.legend(loc='upper right')
plt.show()
