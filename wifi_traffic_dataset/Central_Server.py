import numpy as np
from flask import Flask, request, jsonify
from Model import Net18,data
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CentralServer:
    def __init__(self):
        self.global_model = None
        self.reputation = {}
        self.local_models = {}
        self.aggregation_method = "asynchronous"
 
    def initialize_global_model(self):

       
        # 创建一个模型实例
        model = Net18().to(device)
        data_device  = data.to(device)
        
        # 训练参数
        # 定义损失函数和优化器
        num_epochs = 3
        learning_rate = 0.001
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        if device.type == 'cuda':
            print(f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}")
        else:
            print(f"Using device: {device}")


        # 训练循环
        for epoch in range(num_epochs):
            # 前向传播
            outputs = model(data_device )




            # 计算损失
            loss = criterion(outputs, data_device .y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.global_model = model

        # 保存全局模型到文件
        with open("global_model.pkl", "wb") as file:
            pickle.dump(self.global_model, file)

    def distribute_global_model(self, drone_nodes):

        with open("global_model.pkl", "rb") as file:
            self.global_model = pickle.load(file)
            
        for node in drone_nodes:
            node.receive_global_model(self.global_model)

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation


    def aggregate_models(self):
        if not self.local_models:
            return

        total_reputation = sum(self.reputation.values())
        weighted_models = []

        for drone_id, local_model in self.local_models.items():
            weight = self.reputation[drone_id] / total_reputation
            weighted_model = {k: v * weight for k, v in local_model.items()}
            weighted_models.append(weighted_model)

        # 通过加权平均聚合本地模型
        aggregated_model = {}
        for model in weighted_models:
            for k, v in model.items():
                if k not in aggregated_model:
                    aggregated_model[k] = v
                else:
                    aggregated_model[k] += v
        
        self.global_model = aggregated_model

        # 保存全局模型到文件, 以便下次使用
        with open("global_model.pkl", "wb") as file:
            pickle.dump(self.global_model, file)




    #"0.0.0.0"表示应用程序在所有可用网络接口上运行
    def run(self, port=5000):
        app = Flask(__name__)

        @app.route('/health_check', methods=['GET'])
        def health_check():
            return jsonify({'status': 'OK'})
        
        @app.route('/register', methods=['POST'])
        def register():
            drone_id = request.form['drone_id']
            ip = request.form['ip']
            print("接收到新节点，id："+drone_id+",ip:"+ip)
            # 确定后需要启动新线程，且新线程睡眠5秒等待子节点完成flask初始化
            time.sleep(5)
            print("发送全局模型-执行中--->"+ip)
            try:
                with open("global_model.pkl", "rb") as file:
                    global_model = pickle.load(file)
            except Exception as e:
                print("本地不存在全局模型，训练中……")
                self.initialize_global_model()
                with open("global_model.pkl", "rb") as file:
                    global_model = pickle.load(file)
            # 序列化模型
            global_model_serialized = pickle.dumps(global_model)
            global_model_serialized_base64 = base64.b64encode(global_model_serialized).decode()
            # 发送模型到子节点服务器 
            print("发送全局模型-发送！--->"+ip)
            response = requests.post(f"http://{ip}/receive_model",data={'model': global_model_serialized_base64})
            print("发送全局模型-成功！--->"+ip)
            return jsonify({'status': 'success'})
        

        #序列模型模型
        @app.route('/getModelString', methods=['GET'])
        def getModelString():
            try:
                with open("global_model.pkl", "rb") as file:
                    global_model = pickle.load(file)
            except Exception as e:
                print("本地不存在全局模型，训练中……")
                self.initialize_global_model()
                with open("global_model.pkl", "rb") as file:
                    global_model = pickle.load(file)
            # 序列化模型
            global_model_serialized = pickle.dumps(global_model)
            global_model_serialized_base64 = base64.b64encode(global_model_serialized).decode()
            return jsonify({'model': global_model_serialized_base64})
        
        @app.route('/upload_model', methods=['POST'])
        def upload_model():
            drone_id = request.form['drone_id']
            local_model_serialized = request.form['local_model']
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            local_model_serialized = pickle.dumps(local_model)

            performance = float(request.form['performance'])
            # 更新声誉分数
            self.update_reputation(drone_id, performance)

            # 聚合本地模型并更新全局模型
            #self.aggregate_models({drone_id: local_model})
            print("LOGGER-INFO: received model from child node and updated")
            return jsonify({'status': 'success'})
        
        @app.route('/distribute', methods=['POST'])
        def send_model():
            # TODO 这个之后改成批量的
            url = request.json['url'] # 分发本机的模型到子节点
            
            with open("global_model.pkl", "rb") as file:
                global_model = pickle.load(file)
            # 序列化模型
            global_model_serialized = pickle.dumps(global_model)
            global_model_serialized_base64 = base64.b64encode(global_model_serialized).decode()
            # 发送模型到子节点服务器 
            response = requests.post(f"http://{url}/receive_model",data={'model': global_model_serialized_base64})
            print("LOG-INFO:Global model sent to node:"+url)
            #print("LOG-INFO:Global model data:"+global_model_serialized_base64)
            return jsonify({'status': 'success'})

        app.run(host="localhost", port=port)

central_server_instance = CentralServer()
central_server_instance.run()