import numpy as np
from flask import Flask, request, jsonify
from Model import Net18, data, data_test, Net18_3
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2
data_device = data.to(device)
data_test_device = data_test.to(device)


class CentralServer:
    def __init__(self):
        self.global_model = None
        self.reputation = {}
        self.local_models = {}
        self.aggregation_method = "asynchronous"

    def initialize_global_model(self):
        num_epochs = 1000
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

        self.global_model = model

        # 保存全局模型到文件， 以pt形式保存
        torch.save(self.global_model.state_dict(), "global_model.pt")

    # def distribute_global_model(self, drone_nodes):
    #     # 从文件中加载全局模型
    #     with open("global_model.pkl", "rb") as file:
    #         self.global_model = pickle.load(file)

    #     for node in drone_nodes:
    #         node.receive_global_model(self.global_model)
    def distribute_global_model(self, drone_nodes):
        # 从文件中加载全局模型
        self.global_model = Net18().to(device)
        self.global_model.load_state_dict(torch.load("global_model.pt"))

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

        # 保存全局模型到文件， 以pt形式保存
        torch.save(self.global_model.state_dict(), "global_model.pt")

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
            print("接收到新节点，id：" + drone_id + ",ip:" + ip)
            # 确定后需要启动新线程，且新线程睡眠1秒等待子节点完成flask初始化
            time.sleep(1)
            print("发送全局模型-执行中--->" + ip)
            try:
                self.global_model = Net18(num_output_features).to(device)
                self.global_model.load_state_dict(torch.load("global_model.pt"))
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

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            local_model_serialized = request.form["local_model"]
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            local_model_serialized = pickle.dumps(local_model)

            performance = float(request.form["performance"])
            # 更新声誉分数
            self.update_reputation(drone_id, performance)

            # 聚合本地模型并更新全局模型
            # self.aggregate_models({drone_id: local_model})
            print("LOGGER-INFO: received model from child node and updated")
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
