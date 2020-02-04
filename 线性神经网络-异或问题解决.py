#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt


# In[33]:


#输入数据
X = np.array([[1,0,0,0,0,0],
              [1,0,1,0,0,1],
              [1,1,0,1,0,0],
              [1,1,1,1,1,1]])
#标签
Y = np.array([[-1],
              [1],
              [1],
              [-1]])

#权值初始化，3行1列，取值范围-1到1
W = (np.random.random([6,1])-0.5)*2
print(W)

#学习率设置
lr = 0.11
#计算迭代次数
n = 0
#神经网络输出
O = 0

def update():
    global X,Y,W,lr,n
    n+=1
    O = np.dot(X,W)  #y=x线性激活函数
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C


# In[54]:


for _ in range(1000):
    update()#更新权值
    
#正样本
x1 = [0,1]
y1 = [1,0]
#负样本
x2 = [0,1]
y2 = [0,1]


#画图，计算x2的值，root=1返回正根
def calculate(x,root):
    a=W[5]
    b=W[2]+W[4]*x
    c=W[1]*x+W[3]*x*x+W[0]
    if(root==1):
        return((-b+np.sqrt(b*b-4*a*c))/2/a)
    if(root==0):
        return((-b-np.sqrt(b*b-4*a*c))/2/a)
    

xdata = np.linspace(0,1)

plt.figure()

plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,0),'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')

plt.show()


# In[55]:


#输出,和实际结果相同
print(np.dot(X,W))


# In[ ]:




