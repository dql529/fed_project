import requests
import torch
import torch.nn as nn
import torch.optim as optim

from Model import Net18, edges
import pickle
import io
import base64
from flask import Flask, request, jsonify
from multiprocessing import *
import numpy as np
import sys
import time
from Model import Net18, data, data_test, Net18_3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2
data_device = data.to(device)
data_test_device = data_test.to(device)


class DroneNode:
    def __init__(self):
        self.port = 5001
        self.central_server_ip = "localhost:5000"
        self.drone_id = 1
        self.local_data = None
        self.local_model = None
        self.performance = None

    """DroneNode类在接收到全局模型时,将使用global_model的权重克隆一份副本,并将其分配给self.local_model.
这样，每个无人机节点都可以在本地训练自己的模型副本，并在训练完成后将其上传给中心服务器。
中心服务器可以聚合这些本地模型，从而更新全局模型."""

    def receive_global_model(self, global_model):
        self.global_model = global_model
        self.local_model = Net18(num_output_features).to(device)
        self.local_model.load_state_dict(global_model.state_dict())

    def train_local_model(self, num_epochs=10, batch_size=64, learning_rate=0.001):
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

        self.local_model = Net18(num_output_features).to(device)

        # 定义损失函数和优化器

        num_epochs = 5
        learning_rate = 0.02
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)
        self.local_model = Net18(num_output_features).to(device)

        if self.local_model.num_output_features == 1:
            criterion = nn.BCEWithLogitsLoss()
        elif self.local_model.num_output_features == 2:
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    self.local_model.num_output_features
                )
            )

        if device.type == "cuda":
            print(
                f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
            )
        else:
            print(f"Using device: {device}")

        def compute_loss(outputs, labels):
            if self.local_model.num_output_features == 1:
                return criterion(outputs, labels)
            elif self.local_model.num_output_features == 2:
                return criterion(outputs, labels.squeeze().long())
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

        def to_predictions(outputs):
            if self.local_model.num_output_features == 1:
                return (torch.sigmoid(outputs) > 0.5).float()
            elif self.local_model.num_output_features == 2:
                return outputs.argmax(dim=1)
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

            # Evaluate the self.local_model on the test data

        def evaluate(data_test_device):
            self.local_model.eval()  # Set the self.local_model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = self.local_model(data_test_device)

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

        # 训练循环
        accuracies = []
        for epoch in range(num_epochs):
            self.local_model.train()  # Set the model to training mode
            outputs = self.local_model(data_device)

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
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1

    def upload_local_model(self, central_server_ip):
        # 序列化本地模型
        local_model_serialized = pickle.dumps(self.local_model)
        local_model_serialized_base64 = base64.b64encode(
            local_model_serialized
        ).decode()

        # 评估本地模型性能 ,来自上述训练的 accuracy, precision, recall, f1

        performance = self.accuracy, self.precision, self.recall, self.f1
        print(performance)
        # 发送本地模型及其性能到中心服务器
        response = requests.post(
            f"http://{central_server_ip}/upload_model",
            data={
                "drone_id": self.drone_id,
                "local_model": local_model_serialized_base64,
                "performance": performance,
            },
        )

        print("Response status code:", response.status_code)
        print("Response content:", response.text)

        if response.json()["status"] == "success":
            print(f"Drone {self.drone_id}: Model uploaded successfully.")
        else:
            print(f"Drone {self.drone_id}: Model upload failed.")

    def registerToMaster(self):
        time.sleep(2)
        print("连接到主节点……,本节点端口：" + str(self.port) + "\n")
        response = requests.post(
            f"http://{self.central_server_ip}/register",
            data={"drone_id": str(self.drone_id), "ip": "localhost:" + str(self.port)},
        )
        print("Response status code:", response.status_code)
        print("Response content:", response.text)
        print("主节点连接建立结束……\n")

    def config(self, drone_id, local_data):
        self.drone_id = drone_id
        self.local_data = local_data

    def run(self):
        app = Flask(__name__)

        @app.route("/health_check", methods=["POST"])
        def health_check():
            # drone_id = request.form['drone_id']
            return jsonify({"status": "OK"})

        @app.route("/config", methods=["POST"])
        def config():
            drone_id = request.json["drone_id"]
            # local_data = request.form['local_data']
            self.config(drone_id, data)  # 这里初始化配置，输入初始化数据，可以直接读本地的数据，分离部署也可以把本地数据拷贝
            return jsonify({"status": "初始化配置成功"})

        @app.route("/receive_model", methods=["POST"])
        def receiveModel():
            model_serialized_base64 = request.form["model"]
            model_serialized = base64.b64decode(model_serialized_base64)

            # Load the model's state_dict from the serialized byte stream
            buffer = io.BytesIO(model_serialized)
            state_dict = torch.load(buffer)

            # Create a new model and load the state_dict into it
            model = Net18(num_output_features).to(device)
            model.load_state_dict(state_dict)

            self.receive_global_model(model)
            print("LOGGER-INFO: global model received")

            # Evaluate the received global model
            #
            #
            #
            #
            # 先看看接受的模型准不准确
            def to_predictions(outputs):
                if self.local_model.num_output_features == 1:
                    return (torch.sigmoid(outputs) > 0.5).float()
                elif self.local_model.num_output_features == 2:
                    return outputs.argmax(dim=1)
                else:
                    raise ValueError(
                        "Invalid number of output features: {}".format(
                            self.local_model.num_output_features
                        )
                    )

            self.local_model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = self.local_model(data_test_device)

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
                print(f"Accuracy of received model: {accuracy}")
                print(f"Precision of received model: {precision}")
                print(f"Recall of received model: {recall}")
                print(f"F1 Score of received model: {f1}")
            time.sleep(5)
            print("接收到全局模型，训练中")
            self.train_local_model()
            print("本节点训练完毕")
            print("发送本地训练结果至主节点……")
            self.upload_local_model(self.central_server_ip)
            print("发送完毕……")
            return jsonify({"status": "OK"})

        @app.route("/train", methods=["GET"])
        def train():
            self.train_local_model()
            return jsonify({"status": "train finished"})

        @app.route("/uploadToMaster", methods=["POST"])
        def uploadToMaster(ip=self.central_server_ip):
            ip = request.json["ip"]
            self.upload_local_model(ip)
            return jsonify({"status": "upload to master succeed"})

        app.run(host="localhost", port=self.port)


if __name__ == "__main__":
    drone_node_instance = DroneNode()
    drone_node_instance.port = sys.argv[1]
    drone_node_instance.drone_id = sys.argv[2]
    # 初次连接，接收全局模型，先训练一次
    p1 = Process(target=drone_node_instance.registerToMaster)
    p1.start()
    # drone_node_instance.registerToMaster()
    drone_node_instance.run()
