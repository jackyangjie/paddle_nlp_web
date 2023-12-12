import pandas as pd                  #读取数据用
import numpy as np            #处理数据用
from paddlets import TSDataset       #核心模块
import datetime                      #处理实践戳
from matplotlib import pyplot as plt #绘图必备
import tqdm                          #显示进度
from sklearn.preprocessing import StandardScaler #用于数据标准化

df_train = pd.read_csv(r"/mnt/data/trajectory/train.csv")
tm=df_train['timestamp']

def convert_timestamp(tm):
    d=datetime.datetime.fromtimestamp(tm)
    return d

def get_day(tm):
    d=tm.date()
    # return d.strftime("%Y-%m-%d %H:%M:%S")
    return d

df_train['ts']=tm.apply(lambda x :convert_timestamp(x))   #将timestamp时间转换为长时间格式
df_train['day']=df_train['ts'].apply(lambda x :get_day(x))#提取以日期用以区分训练集



dup_mmsi = list(df_train['mmsi'].drop_duplicates())  #mmsi索引列表
df_combin=[]                                                            #缓存处理数据集，用于事后合并
for mmsi_index in tqdm.tqdm(dup_mmsi):                                  #按照mmsi索引不同船只
    dft= df_train[df_train['mmsi']==mmsi_index]
    dup_day = list(dft['day'].drop_duplicates())                        #区分不同日期划分子集
    if len(dup_day)>0:
        for days in dup_day:
            dd = dft[dft['day']==days]
            dd.set_index("ts", inplace = True)
            new_df_train=dd.asfreq("1s").interpolate(method='slinear')     #以1秒为单位进行线性差值
            new_df_train['ts']=new_df_train.index
            new_df_train['day']=days
            df_combin.append(new_df_train)
            # new_df_train.to_excel(r'./train_split/'+str(mmsi_index)+"_"+str(days)+".xlsx")#将数据保存到本地

df_last=pd.concat(df_combin) #将处理后的数据进行合并


fig,ax=plt.subplots()
sc=ax.scatter(df_last.iloc[:,2],df_last.iloc[:,1],s=0.2,c=df_last.iloc[:,0])
ax.set_xlabel("lon")
ax.set_ylabel("lat")
ax.set_title("Train data")

target_data=df_last[df_last['mmsi']==1]
target_days = list(target_data['day'].drop_duplicates())
target_data=df_last[df_last['day']==target_days[1]]



ss=StandardScaler()
data = ss.fit_transform(target_data[['lat','lon']])
target_data[['lat','lon']]=data

#使用TSDataset的方法，对数据信息模型适配
dataset = TSDataset.load_from_dataframe(
    target_data,
    time_col='ts',              #指定ts列为时间列
    target_cols=['lat','lon'],
    freq='1s'
)
dataset.plot()

train_dataset, val_test_dataset = dataset.split(0.7)
val_dataset, test_dataset = val_test_dataset.split(0.5)
train_dataset.plot()


from paddlets.models.forecasting import LSTNetRegressor
import paddle
import paddle.nn.functional as F
from paddlets.metrics import MSE
mse=MSE()
lstnet= LSTNetRegressor(
    in_chunk_len = 10,  #训练步长为10秒
    out_chunk_len = 1,  #预测量为未来1秒的数据
    optimizer_fn=paddle.optimizer.AdamW,  # 设置优化器
    optimizer_params=dict(learning_rate=3e-4),  # 设置学习率
    eval_metrics =["mse"],#设置评估指标为MSE
    max_epochs=100
)

Train_again = True  #定义开关变量 如果需要重新训练变量，则设置为TRUE
if Train_again:
    lstnet.fit(train_dataset, val_dataset) #重新训练模型
    lstnet.save("model/LSTNet"+str(datetime.datetime.now())+".model")               #保存模型
else:                                  #如果无需训练，则读取先前训练好的模型用于预测
    lstnet.load("model/LSTNet.model")