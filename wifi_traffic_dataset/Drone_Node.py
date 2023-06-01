import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Model import LSTMModel
import pickle
from data_loader import load_dataset, load_local_dataset
import base64
from flask import Flask, request, jsonify
from multiprocessing import *
from Dataset import features, labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
import time

class DroneNode:
    def __init__(self):
        self.port = 5002
        self.central_server_ip = "localhost:5000"
        self.drone_id = 2
        self.local_data = None
        self.local_model = None
        self.performance = None
    '''DroneNode类在接收到全局模型时,将使用global_model的权重克隆一份副本,并将其分配给self.local_model.
这样，每个无人机节点都可以在本地训练自己的模型副本，并在训练完成后将其上传给中心服务器。
中心服务器可以聚合这些本地模型，从而更新全局模型.'''
    def receive_global_model(self, global_model):
        # 克隆全局模型的权重
        self.global_model = global_model
        self.local_model = LSTMModel(global_model.input_size, 
                                      global_model.hidden_size, 
                                      global_model.num_layers, 
                                      global_model.output_size)
        self.local_model.load_state_dict(global_model.state_dict())

   

    def train_local_model(self, num_epochs=10, batch_size=64, learning_rate=0.001):
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

        self.local_model.to(device)
        local_loader = load_local_dataset()

        # 定义损失函数和优化器
        num_epochs = 3
        learning_rate = 0.001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)


        if device.type == 'cuda':
            print(f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}")
        else:
            print(f"Using device: {device}")
        # 训练循环
        for epoch in range(num_epochs):
            for data, targets in local_loader:
                # 将数据移动到GPU上
                data, targets = data.to(device), targets.to(device)
                 # 初始化隐藏状态并将其移至GPU
                h0 = torch.zeros(self.local_model.num_layers, data.size(0), self.local_model.hidden_size).to(device)
                c0 = torch.zeros(self.local_model.num_layers, data.size(0), self.local_model.hidden_size).to(device)
                # 前向传播
                outputs = self.local_model(data)
                targets = targets.unsqueeze(1)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Drone {self.drone_id} - Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
        print("Drone {}: Local model training complete.".format(self.drone_id))
        

    def evaluate_local_model(self):
        # 评估本地模型的性能（此处仅作示例，实际情况下可能需要根据其他指标计算性能）
        #self.performance = ... 根据实际情况完成性能评估代码

        performance = 0
        return performance

    def upload_local_model(self, central_server_ip):

        
        # 序列化本地模型
        local_model_serialized = pickle.dumps(self.local_model)
        local_model_serialized_base64 = base64.b64encode(local_model_serialized).decode()

        # 评估本地模型性能
        performance = self.evaluate_local_model()

        # 发送本地模型及其性能到中心服务器
        response = requests.post(f"http://{central_server_ip}/upload_model",
                         data={'drone_id': self.drone_id, 'local_model': local_model_serialized_base64, 'performance': performance})



        print("Response status code:", response.status_code)
        print("Response content:", response.text)

        if response.json()['status'] == 'success':
            print(f"Drone {self.drone_id}: Model uploaded successfully.")
        else:
            print(f"Drone {self.drone_id}: Model upload failed.")

    def registerToMaster(self):
        time.sleep(3)
        print("连接到主节点……,本节点端口："+str(self.port)+"\n")
        response = requests.post(f"http://{self.central_server_ip}/register",
                         data={'drone_id': str(self.drone_id),"ip":"localhost:"+str(self.port)})
        print("Response status code:", response.status_code)
        print("Response content:", response.text)
        print("主节点连接建立结束……\n")

    def config(self,drone_id,local_data):
        self.drone_id = drone_id
        self.local_data = local_data

    def run(self):
        app = Flask(__name__)

        @app.route('/health_check', methods=['POST'])
        def health_check():
            #drone_id = request.form['drone_id']
            return jsonify({'status': 'OK'})
        
        @app.route('/config', methods=['POST'])
        def config():
            drone_id = request.json['drone_id']
            #local_data = request.form['local_data']
            self.config(drone_id,features) # 这里初始化配置，输入初始化数据，可以直接读本地的数据，分离部署也可以把本地数据拷贝
            return jsonify({'status': '初始化配置成功'})

        @app.route('/receive_model', methods=['POST'])
        def receiveModel():
            model_serialized = request.form['model']
            model = pickle.loads(base64.b64decode(model_serialized))
            self.receive_global_model(model)
            print("LOGGER-INFO: global model received")
            print("接收到全局模型，训练中")
            self.train_local_model()
            print("本节点训练完毕")
            print("发送本地训练结果至主节点……")
            self.upload_local_model(self.central_server_ip)
            print("发送完毕……")
            return jsonify({'status': 'OK'})
        
        @app.route('/train', methods=['GET'])
        def train():
            self.train_local_model()
            return jsonify({'status': 'train finished'})
        
        @app.route('/uploadToMaster', methods=['POST'])
        def uploadToMaster(ip=self.central_server_ip):
            ip = request.json['ip'] 
            self.upload_local_model(ip)
            return jsonify({'status': 'upload to master succeed'})
        
        app.run(host="localhost", port=self.port)

if __name__ == '__main__':
    drone_node_instance = DroneNode()
    drone_node_instance.port = sys.argv[1]
    drone_node_instance.drone_id = sys.argv[2]
    # 初次连接，接收全局模型，先训练一次
    p1 = Process(target=drone_node_instance.registerToMaster)
    p1.start()
    # drone_node_instance.registerToMaster()
    drone_node_instance.run()