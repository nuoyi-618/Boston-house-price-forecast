from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib as mpl
from sklearn.metrics import r2_score as r2
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

house_dataset = datasets.load_boston()    #加载波士顿房价数据集
house_data=house_dataset.data  #加载房屋属性参数
house_price = house_dataset.target  #加载房屋均价
df = pd.DataFrame(data=house_data,columns=house_dataset.feature_names)
df['MEDV'] = house_dataset.target
#pfr = pp.ProfileReport(df) #数据描述报告
#pfr.to_file("./example.html")
x_train,x_test,y_train,y_test=train_test_split(house_data,house_price,test_size=0.2)

#数据标准化
scaler = StandardScaler() #f(x)=(x-平均值)/标准差
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#线性回归
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train, y_train)
y_predict_LR = LR.predict(x_test)
r2_score_LR = r2(y_test, y_predict_LR)
print(r2_score_LR)

t = np.arange(len(y_predict_LR))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', lw=2, label='real')
plt.plot(t, y_predict_LR, 'g-', lw=2, label='predict')
plt.legend(loc='best')     
plt.title('LinearRegression', fontsize=18)
plt.xlabel('id', fontsize=15)
plt.ylabel('price', fontsize=15)
plt.grid()
#plt.show()
plt.savefig("LinearRegression.png")
#SVR
svr = SVR(kernel='rbf')                 #构建基于rbf（径向基函数）的SVR模型
svr.fit(x_train,y_train)                #将训练组数据输入进行训练
y_predict_SVR = svr.predict(x_test)    #将处理过的预测组数据输入进行预测，得出结果
#将实际结果与预测结果对比观察，2列的数组，左边列是实际结果，右边列是预测结果
r2_score_SVR = r2(y_test, y_predict_SVR)
result = np.hstack((y_test.reshape(-1,1),y_predict_SVR.reshape(-1,1)))
print(r2_score_SVR)

t = np.arange(len(y_predict_SVR))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', lw=2, label='real')
plt.plot(t, y_predict_SVR, 'g-', lw=2, label='predict')
plt.legend(loc='best')      
plt.title('svr', fontsize=18)
plt.xlabel('id', fontsize=15)
plt.ylabel('price', fontsize=15)
plt.grid()
#plt.show()
plt.savefig("svr.png")