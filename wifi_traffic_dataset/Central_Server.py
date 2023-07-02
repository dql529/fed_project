import numpy as np
from flask import Flask, request, jsonify
from Model import Net18
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import copy
import os
from queue import Queue
from aggregation_solution import weighted_average_aggregation, average_aggregation

os.chdir("C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset")

torch.manual_seed(0)
# 读取数据
server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2

# 写法和drone node.py有区别   Drone node 中定义在clss中，根据self来调用，此处为了方便，直接定义在函数中
data_device = server_train.to(device)
data_test_device = server_test.to(device)


class CentralServer:
    def __init__(self):
        self.global_model = None
        self.reputation = {}
        self.local_models = {}
        self.local_models = Queue()  # 使用队列来存储上传的模型
        self.aggregation_method = "asynchronous"
        self.drone_nodes = {}

    def initialize_global_model(self):
        torch.manual_seed(0)
        num_epochs = 1000
        num_output_features = 2
        learning_rate = 0.02
        model = Net18(num_output_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if model.num_output_features == 1:
            criterion = nn.BCEWithLogitsLoss()
        elif model.num_output_features == 2:
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    model.num_output_features
                )
            )

        if device.type == "cuda":
            print(
                f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
            )
        else:
            print(f"Using device: {device}")

        def compute_loss(outputs, labels):
            if model.num_output_features == 1:
                return criterion(outputs, labels)
            elif model.num_output_features == 2:
                return criterion(outputs, labels.squeeze().long())
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        model.num_output_features
                    )
                )

        def to_predictions(outputs):
            if model.num_output_features == 1:
                return (torch.sigmoid(outputs) > 0.5).float()
            elif model.num_output_features == 2:
                return outputs.argmax(dim=1)
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        model.num_output_features
                    )
                )

            # Evaluate the model on the test data

        def evaluate(data_test_device):
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = model(data_test_device)

                predictions_test = to_predictions(outputs_test)

                # Calculate metrics
                accuracy = accuracy_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
                f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
                return accuracy, precision, recall, f1

        # 训练模型并记录每个epoch的准确率
        accuracies = []
        best_accuracy = 0.0
        best_model_state_dict = None
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            outputs = model(data_device)

            loss = compute_loss(outputs, data_device.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate
            accuracy, precision, recall, f1 = evaluate(data_test_device)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Loss: {loss.item()}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state_dict = copy.deepcopy(model.state_dict())
        # After training, load the best model weights
        model.load_state_dict(best_model_state_dict)
        # Save the best model to a file
        torch.save(model.state_dict(), "global_model.pt")
        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_epoch = accuracies.index(max_accuracy) + 1

        # Print the coordinates of the maximum point
        print(
            f"learning rate {learning_rate}, epoch {num_epochs} and dimension {model.num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
        )

    def distribute_global_model(self, drone_nodes):
        # 从文件中加载全局模型
        self.global_model = Net18(num_output_features).to(device)
        self.global_model.load_state_dict(torch.load("global_model.pt"))

        for node in drone_nodes:
            node.receive_global_model(self.global_model)

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation

    def aggregate_models(self, models_to_aggregate):
        if self.aggregation_method == "asynchronous":
            aggregated_model = weighted_average_aggregation(
                models_to_aggregate, self.reputation
            )
        else:  # 默认使用平均聚合
            aggregated_model = average_aggregation(models_to_aggregate)

        self.global_model = aggregated_model
        # 保存全局模型到文件， 以pt形式保存
        torch.save(self.global_model, "global_model.pt")

    def send_model(self, ip):
        # 加载全局模型
        try:
            self.global_model = Net18(num_output_features).to(device)
            # 先生成模型，再加载训练好的模型
            self.global_model.load_state_dict(torch.load("global_model.pt"))
            print("本地存在全局模型，发送中…… ")
        except Exception as e:
            print("本地不存在全局模型，训练中……")
            self.initialize_global_model()
            self.global_model = Net18(num_output_features).to(device)
            self.global_model.load_state_dict(torch.load("global_model.pt"))

        # Save the model's state_dict to a BytesIO buffer
        buffer = io.BytesIO()
        torch.save(self.global_model.state_dict(), buffer)

        # Get the contents of the buffer
        global_model_serialized = buffer.getvalue()

        # Encode the bytes as a base64 string
        global_model_serialized_base64 = base64.b64encode(
            global_model_serialized
        ).decode()
        # 发送模型到子节点服务器
        print("发送全局模型-发送！--->" + ip)
        response = requests.post(
            f"http://{ip}/receive_model",
            data={"model": global_model_serialized_base64},
        )
        print("发送全局模型-成功！--->" + ip)
        return jsonify({"status": "success"})

    # "0.0.0.0"表示应用程序在所有可用网络接口上运行
    def run(self, port=5000):
        app = Flask(__name__)

        @app.route("/health_check", methods=["GET"])
        def health_check():
            return jsonify({"status": "OK"})

        @app.route("/register", methods=["POST"])
        def register():
            drone_id = request.form["drone_id"]
            ip = request.form["ip"]
            print("接收到新节点,id:" + drone_id + ",ip:" + ip)
            # 将新的无人机节点添加到字典中
            self.drone_nodes[drone_id] = ip

            print("发送全局模型-执行中--->" + ip)
            try:
                self.global_model = Net18(num_output_features).to(device)
                # 先生成模型，再加载训练好的模型
                self.global_model.load_state_dict(torch.load("global_model.pt"))
            except Exception as e:
                print("本地不存在全局模型，训练中……")
                self.initialize_global_model()
                self.global_model = Net18(num_output_features).to(device)
                self.global_model.load_state_dict(torch.load("global_model.pt"))

            self.send_model(ip)
            return jsonify({"status": "success"})  # 添加这一行

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            local_model_serialized = request.form["local_model"]
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            local_model_serialized = pickle.dumps(local_model)

            performance = float(request.form["performance"])
            # 更新声誉分数
            self.update_reputation(drone_id, performance)

            # 将模型添加到队列中
            self.local_models.put({drone_id: local_model})
            print(
                f"Received model from drone {drone_id}, now have {self.local_models.qsize()} models in queue"
            )
            # 检查队列中是否有足够的模型进行聚合
            if self.local_models.qsize() >= 3:
                models_to_aggregate = []
                for _ in range(3):
                    models_to_aggregate.append(self.local_models.get())
                # 聚合本地模型并更新全局模型
                self.aggregate_models(models_to_aggregate)
                print("LOGGER-INFO: received model from child node and updated")

                for drone_id, ip in self.drone_nodes.items():
                    self.send_model(ip)

            return jsonify({"status": "success"})

        @app.route("/distribute", methods=["POST"])
        def send_model():
            # TODO 这个之后改成批量的
            url = request.json["url"]  # 分发本机的模型到子节点

            with open("global_model.pkl", "rb") as file:
                global_model = pickle.load(file)
            # 序列化模型
            global_model_serialized = pickle.dumps(global_model)
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()
            # 发送模型到子节点服务器
            response = requests.post(
                f"http://{url}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            print("LOG-INFO:Global model sent to node:" + url)
            # print("LOG-INFO:Global model data:"+global_model_serialized_base64)
            return jsonify({"status": "success"})

        app.run(host="localhost", port=port)


central_server_instance = CentralServer()
central_server_instance.run()
