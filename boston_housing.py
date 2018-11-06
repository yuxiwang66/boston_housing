# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:35:53 2018

@author: yuxi
"""
#加载波士顿房价数据集
from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#准备数据，标准化
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#模型定义
from keras import models
from keras import layers
def build_model():#下面需要将同一个模型多次实例化，所以用一个函数来构建模型
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model


#K折验证
import numpy as np
k=4
num_val_samples=len(train_data)//k

# =============================================================================
# #设置100轮次
# num_epochs=100
# all_scores=[]
# 
# for i in range(k):
#     print('processing fold #',i)
#     val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
#     val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
#     
#     partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
#     partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
#     model =build_model()
#     model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0)
#     val_mse,val_mae=model.evaluate(val_data,val_targets,verbose=0)
#     all_scores.append(val_mae)
# =============================================================================

#设置500轮次
num_epochs=500
all_mae_histories=[]
for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    model =build_model()
    history=model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs,batch_size=1,verbose=0)
    mae_history=history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
#计算所有轮次中的K折验证分数平均值
average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#绘制验证分数
import matplotlib.pyplot as plt
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#绘制验证分数（删除前10个点，并做光滑处理）
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous =smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history =smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#训练最终模型
model=build_model()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
print(test_mae_score)





