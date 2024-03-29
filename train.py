# -*- coding:utf-8 -*-

# 导入panda，keras 和tensorflow
import pandas as pd
from tensorflow.keras.models import Sequential #顺序模型
from tensorflow.keras.layers import Dense #全链接层


# 加载样本数据集，划分为x和y DataFrame
df = pd.read_csv("https://github.com/bgweber/Twitch/raw/master/Recommendations/games-expand.csv")

df_data = df.drop(['label'], axis=1)
df_labl = df['label']

train_data = df_data.loc[0:20000]
train_targets = df_labl.loc[0:20000]

test_data = df_data.loc[20001:]
test_targets = df_labl.loc[20001:]


# 训练次数
epochs = 200

# 搭建模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])   #优化器选择的是Adam，损失函数为MSE函数

# 训练模型
history = model.fit(train_data, train_targets,
                    batch_size=32,                      #批次大小为32
                    epochs=epochs,                     #循环次数为 之前定义的200
                    validation_data=(test_data, test_targets),  #验证集
                    shuffle=True)                      #打乱标签


# 以H5格式保存模型
model.save("model.h5")

