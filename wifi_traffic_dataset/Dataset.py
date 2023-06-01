
import pandas as pd
from sklearn.model_selection import train_test_split
# file_path_tr1 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\1\\data_tr.csv'
# file_path_te1 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\1\\data_te.csv'
# # 读取数据
# data_tr_1 = pd.read_csv(file_path_tr1, sep=' ',header=None)
# data_te_1 = pd.read_csv(file_path_te1, sep=' ',header=None)

# data_1 = pd.concat([data_tr_1,data_te_1],axis=1)
# data_transposed1 = data_1.T
# #写入csv文件
# #df1 = pd.DataFrame(data_transposed1)


df = pd.read_csv('C:\\Users\\ROG\\Desktop\\UAV_Project\\data1.csv', sep=' ')
# 按9：1 划分训练，测试集
train_df, test_df = train_test_split(df, test_size=0.1, random_state=0)
features = test_df.iloc[:,:54]
labels = test_df.iloc[:,54]



'''
df1.to_csv('data1.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中


file_path_tr2 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\2\\data_tr.csv'
file_path_te2 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\2\\data_te.csv'
# 读取数据
data_tr_2 = pd.read_csv(file_path_tr2, sep=' ',header=None)
data_te_2 = pd.read_csv(file_path_te2, sep=' ',header=None)

data_2 = pd.concat([data_tr_2,data_te_2],axis=1)
data_transposed2 = data_2.T
#写入csv文件
df2 = pd.DataFrame(data_transposed2)
df2.to_csv('data2.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中


file_path_tr3 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\3\\data_tr.csv'
file_path_te3 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\3\\data_te.csv'
# 读取数据
data_tr_3 = pd.read_csv(file_path_tr3, sep=' ',header=None)
data_te_3 = pd.read_csv(file_path_te3, sep=' ',header=None)

data_3 = pd.concat([data_tr_3,data_te_3],axis=1)
data_transposed3 = data_3.T
#写入csv文件
df3 = pd.DataFrame(data_transposed3)
df3.to_csv('data3.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中


file_path_tr4 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\4\\data_tr.csv'
file_path_te4 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\4\\data_te.csv'
# 读取数据
data_tr_4 = pd.read_csv(file_path_tr4, sep=' ',header=None)
data_te_4 = pd.read_csv(file_path_te4, sep=' ',header=None)

data_4 = pd.concat([data_tr_4,data_te_4],axis=1)
data_transposed4 = data_4.T
#写入csv文件
df4= pd.DataFrame(data_transposed4)
df4.to_csv('data4.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中


file_path_tr5 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\5\\data_tr.csv'
file_path_te5 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\5\\data_te.csv'
# 读取数据
data_tr_5 = pd.read_csv(file_path_tr5, sep=' ',header=None)
data_te_5 = pd.read_csv(file_path_te5, sep=' ',header=None)

data_5 = pd.concat([data_tr_5,data_te_5],axis=1)
data_transposed5 = data_5.T
#写入csv文件
df5 = pd.DataFrame(data_transposed5)
df5.to_csv('data5.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中



file_path_tr6 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\6\\data_tr.csv'
file_path_te6 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\6\\data_te.csv'
# 读取数据
data_tr_6 = pd.read_csv(file_path_tr6, sep=' ',header=None)
data_te_6 = pd.read_csv(file_path_te6, sep=' ',header=None)

data_6= pd.concat([data_tr_6,data_te_6],axis=1)

data_transposed6 = data_6.T
#写入csv文件
df6 = pd.DataFrame(data_transposed6)
df6.to_csv('data6.csv', sep = ' ',index=False)  # index=False表示不将索引写入文件中
'''
'''
查看方式
file_path = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\data1.csv'
data = pd.read_csv(file_path, sep=' ')
print(data)
'''









'''
file_path_H1 = 'C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset\\1\\H.csv'
data_h = pd.read_csv(file_path_H1, sep=' ',header=None)
'''




