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
import threading
import json
from tools import plot_accuracy_vs_epoch
from matplotlib import pyplot as plt
import logging
import sys

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
os.chdir("C:\\Users\\ROG\\Desktop\\UAV_Project\\wifi_traffic_dataset")

torch.manual_seed(0)
# 读取数据
server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")
num_output_features = 2
if num_output_features == 1:
    criterion = nn.BCEWithLogitsLoss()
elif num_output_features == 2:
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(
        "Invalid number of output features: {}".format(num_output_features)
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2

# 写法和drone node.py有区别   Drone node 中定义在clss中，根据self来调用，此处为了方便，直接定义在函数中
data_device = server_train.to(device)
data_test_device = server_test.to(device)


class CentralServer:
    def __init__(self):
        self.global_model = None
        self.aggregated_global_model = None
        self.reputation = {}
        self.local_models = {}
        self.local_models = Queue()  # 使用队列来存储上传的模型
        self.lock = threading.Lock()  # 创建锁
        self.aggregation_method = "average aggregation"
        # self.aggregation_method = "average aggregation"

        self.drone_nodes = {}
        self.aggregation_accuracies = []
        self.num_aggregations = 0  # 记录聚合次数
        threading.Thread(target=self.check_and_aggregate_models).start()

    def fed_evaluate(self, model, data_test_device):
        with torch.no_grad():  # Do not calculate gradients to save memory
            outputs_test = model(data_test_device)

            predictions_test = self.to_predictions(outputs_test)

            # Calculate metrics
            accuracy = accuracy_score(data_test_device.y.cpu(), predictions_test.cpu())
            precision = precision_score(
                data_test_device.y.cpu(), predictions_test.cpu()
            )
            recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
            f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
            return accuracy, precision, recall, f1

    def compute_loss(self, outputs, labels):
        if self.global_model.num_output_features == 1:
            return criterion(outputs, labels)
        elif self.global_model.num_output_features == 2:
            return criterion(outputs, labels.squeeze().long())
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    self.global_model.num_output_features
                )
            )

    def to_predictions(self, outputs):
        if self.global_model.num_output_features == 1:
            return (torch.sigmoid(outputs) > 0.5).float()
        elif self.global_model.num_output_features == 2:
            return outputs.argmax(dim=1)
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    self.global_model.num_output_features
                )
            )

    def check_and_aggregate_models(self):
        print("LOGGER-INFO: check_and_aggregate_models() is called")
        # 检查队列中是否有足够的模型进行聚合
        start_time = time.time()
        max_wait_time = 50  # 最大等待时间（秒）
        while True:
            if self.local_models.qsize() >= 3:
                models_to_aggregate = []
                self.lock.acquire()
                try:
                    for _ in range(3):
                        models_to_aggregate.append(self.local_models.get())
                finally:
                    self.lock.release()
                self.aggregate_models(models_to_aggregate)

                for drone_id, ip in self.drone_nodes.items():
                    self.send_model(ip, "self.aggregated_global_model")

                time.sleep(5)
                accuracy, precision, recall, f1 = self.fed_evaluate(
                    self.aggregated_global_model, data_test_device
                )
                self.aggregation_accuracies.append(accuracy)
                self.num_aggregations += 1  # 增加模型聚合的次数
                print("Current aggregation accuracy: ", accuracy)

            # 当有10条记录就开始动态画图
            if self.num_aggregations == 30:
                plot_accuracy_vs_epoch(
                    self.aggregation_accuracies,
                    self.num_aggregations,
                    learning_rate=0.02,
                )
                plt.savefig(
                    f"accuracy_vs_epoch_{self.num_aggregations}_aggregration method {self.aggregation_method}.png"
                )
                print("Aggregation accuracies so far: ", self.aggregation_accuracies)
                print("Program is about to terminate")
                sys.exit()  # Terminate the program

            # 如果等待时间超过最大等待时间，退出循环
            if time.time() - start_time > max_wait_time:
                print("Max wait time exceeded, exiting")
                break

    def initialize_global_model(self):
        torch.manual_seed(0)
        num_epochs = 10
        num_output_features = 2
        learning_rate = 0.02
        model = Net18(num_output_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if device.type == "cuda":
            print(
                f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
            )
        else:
            print(f"Using device: {device}")

        def evaluate(data_test_device):
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = model(data_test_device)

                predictions_test = self.to_predictions(outputs_test)

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

            loss = self.compute_loss(outputs, data_device.y)

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

        self.global_model = model

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation

    def aggregate_models(self, models_to_aggregate):
        if self.aggregation_method == "async_weighted_average":
            print("Using weighted average aggregation")
            aggregated_model = weighted_average_aggregation(
                models_to_aggregate, self.reputation
            )
        elif self.aggregation_method == "average aggregation":
            print("Using average aggregation")
            aggregated_model = average_aggregation(models_to_aggregate)
        else:  # 默认使用平均聚合
            print("Using average aggregation")
            aggregated_model = average_aggregation(models_to_aggregate)

        self.aggregated_global_model = Net18(num_output_features).to(device)  # 创建新的模型实例
        self.aggregated_global_model.load_state_dict(aggregated_model)  # 加载聚合后的权重

        # 保存全局模型到文件， 以pt形式保存
        torch.save(
            self.aggregated_global_model.state_dict(), "aggregrated_global_model.pt"
        )

    def send_model(self, ip, model_type="global_model"):
        # 根据 model_type 加载不同的模型
        if model_type == "aggregated_global_model":
            model_path = "aggregated_global_model.pt"
        else:  # 默认加载 global_model
            model_path = "global_model.pt"

        # 加载模型
        try:
            self.global_model = Net18(num_output_features).to(device)
            # 先生成模型，再加载训练好的模型
            self.global_model.load_state_dict(torch.load(model_path))
            print(f"本地存在 {model_type}，发送中…… ")
        except Exception as e:
            print(f"本地不存在 {model_type}，训练中……")
            self.initialize_global_model()

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
        return json.dumps({"status": "success"})

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
            self.send_model(ip, "global_model")
            return jsonify({"status": "success"})

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            local_model_serialized = request.form["local_model"]
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            local_model_serialized = pickle.dumps(local_model)

            performance = float(request.form["performance"])
            # 更新声誉分数
            self.update_reputation(drone_id, performance)
            print("reputation: " + str(self.reputation))

            self.local_models.put({drone_id: local_model})
            print("local models: " + str(self.local_models))
            return jsonify({"status": "success"})

        app.run(host="localhost", port=port)


central_server_instance = CentralServer()
central_server_instance.run()
