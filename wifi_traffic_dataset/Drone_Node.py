import requests
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Net18
import pickle
import io
import base64
from flask import Flask, request, jsonify
from multiprocessing import *
import numpy as np
import sys
import time
from Model import Net18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from Dataset import n_nodes

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2


# 测试用
# 读取数据
server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")


# 对于每个节点，加载训练和测试数据
node_train = []
node_test = []
for i in range(1, n_nodes + 1):
    node_train.append(torch.load(f"data_object/node_train_{i}.pt"))
    node_test.append(torch.load(f"data_object/node_test_{i}.pt"))


class DroneNode:
    def __init__(self, local_train_data, local_test_data):
        self.port = 5001
        self.central_server_ip = "localhost:5000"
        self.drone_id = 1
        self.local_model = None
        self.performance = None
        self.data_device = local_train_data.to(device)
        # self.data_test_device = local_test_data.to(device)
        # 下面这行测试用
        self.data_test_device = server_test.to(device)

    """DroneNode类在接收到全局模型时,将使用global_model的权重克隆一份副本,并将其分配给self.local_model.
这样，每个无人机节点都可以在本地训练自己的模型副本，并在训练完成后将其上传给中心服务器。
中心服务器可以聚合这些本地模型，从而更新全局模型."""

    def receive_global_model(self, global_model):
        self.global_model = global_model
        self.local_model = Net18(num_output_features).to(device)
        self.local_model.load_state_dict(global_model.state_dict())

    def train_local_model(self):
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

        num_epochs = 1
        learning_rate = 0.0001

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)

        # self.local_model = Net18(num_output_features).to(device)

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
            outputs = self.local_model(self.data_device)

            loss = compute_loss(outputs, self.data_device.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate
            accuracy, precision, recall, f1 = evaluate(self.data_test_device)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Loss: {loss.item()}")
            print(f"Accuracy: {accuracy}")
            # print(f"Precision: {precision}")
            # print(f"Recall: {recall}")
            # print(f"F1 Score: {f1}")
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1

        def to_percent(y, position):
            return f"{100*y:.2f}%"

        formatter = FuncFormatter(to_percent)

        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.plot(
            range(1, num_epochs + 1),
            accuracies,
            marker="o",
            linestyle="-",
            color="b",
            label="Accuracy",
        )  # Plot accuracy
        plt.xlabel("Epoch", fontsize=14)  # Set the label for the x-axis
        plt.ylabel("Accuracy", fontsize=14)  # Set the label for the y-axis
        plt.title("Accuracy vs. Epoch", fontsize=16)  # Set the title
        plt.grid(True)  # Add grid lines
        plt.legend(fontsize=12)  # Add a legend
        plt.xticks(fontsize=12)  # Set the size of the x-axis ticks
        plt.yticks(fontsize=12)  # Set the size of the y-axis ticks
        plt.gca().yaxis.set_major_formatter(
            formatter
        )  # Set the formatter for the y-axis
        plt.xlim([1, num_epochs])  # Set the range of the x-axis
        plt.ylim([0, 1])  # Set the range of the y-axis

        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_epoch = accuracies.index(max_accuracy) + 1

        # Print the coordinates of the maximum point
        print(
            f"learning rate {learning_rate}, epoch {num_epochs} and dimension {num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
        )

        # Annotate the maximum point
        plt.annotate(
            f"Max Accuracy: {100*max_accuracy:.2f}%",
            xy=(max_epoch, max_accuracy),
            xytext=(max_epoch + 5, max_accuracy - 0.1),
            arrowprops=dict(facecolor="red", shrink=0.05),
        )

        plt.show()

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
        print("连接到主节点……,本节点端口：" + str(self.port) + "\n" + "本节点ID：" + str(self.drone_id))
        response = requests.post(
            f"http://{self.central_server_ip}/register",
            data={"drone_id": str(self.drone_id), "ip": "localhost:" + str(self.port)},
        )
        print("Response status code:", response.status_code)
        print("Response content:", response.text)
        print("主节点连接建立结束……\n")

    # def config(self, drone_id, local_data):
    #     self.drone_id = drone_id
    #     self.local_data = local_data

    def run(self):
        app = Flask(__name__)

        @app.route("/health_check", methods=["POST"])
        def health_check():
            # drone_id = request.form['drone_id']
            return jsonify({"status": "OK"})

        # @app.route("/config", methods=["POST"])
        # def config():
        #     drone_id = request.json["drone_id"]
        #     # local_data = request.form['local_data']
        #     self.config(drone_id, data)  # 这里初始化配置，输入初始化数据，可以直接读本地的数据，分离部署也可以把本地数据拷贝
        #     return jsonify({"status": "初始化配置成功"})

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

            # 这边没有问题,global model这个方法将模型分配给了local_local model

            self.local_model.eval()  # Set the model to evaluation mode

            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = self.local_model(self.data_test_device)
                print("outputs_test", outputs_test)
                time.sleep(3)
                predictions_test = to_predictions(outputs_test)
                print("predictions_test", predictions_test)
                time.sleep(3)
                # Calculate metrics
                accuracy = accuracy_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                f1 = f1_score(self.data_test_device.y.cpu(), predictions_test.cpu())
                print(f"Accuracy of received model: {accuracy}")
                print(f"Precision of received model: {precision}")
                print(f"Recall of received model: {recall}")
                print(f"F1 Score of received model: {f1}")

            time.sleep(3)
            print("接收到全局模型，训练中")
            self.train_local_model()
            print("本节点训练完毕")
            print("发送本地训练结果至主节点……")
            self.upload_local_model(self.central_server_ip)
            print("发送完毕……")

            return jsonify({"status": "OK"})

            # os._exit(0)

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
    drone_id = int(sys.argv[2])
    drone_node_instance = DroneNode(node_train[drone_id - 1], node_test[drone_id - 1])
    drone_node_instance.port = sys.argv[1]
    drone_node_instance.drone_id = drone_id

    time.sleep(2)
    # 初次连接，接收全局模型，先训练一次
    p1 = Process(target=drone_node_instance.registerToMaster)
    p1.start()
    # drone_node_instance.registerToMaster()
    drone_node_instance.run()
