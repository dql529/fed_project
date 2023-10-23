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
from tools import plot_accuracy_vs_epoch, sigmoid, exponential_decay
from matplotlib import pyplot as plt
import logging
import sys
from web3 import Web3

# 读取整个JSON文件
with open("blockchain/build/contracts/ModelRegistry.json", "r", encoding="utf-8") as f:
    data = json.load(f)
abi = data["abi"]

torch.manual_seed(0)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
learning_rate = 0.01
num_output_features = 2
criterion = nn.CrossEntropyLoss()

# 读取数据
server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.aggregation_method = "async weighted  aggregation"
        self.new_model_event = threading.Event()
        self.drone_nodes = {}
        self.aggregation_accuracies = []
        self.num_aggregations = 0  # 记录聚合次数
        self.data_age = {}
        self.low_performance_counts = {}
        # import blockchain module
        self.w3 = Web3(
            Web3.HTTPProvider("http://localhost:7545")
        )  # Replace with your Ethereum node address
        self.contract_address = "0x6F7159033b9674bD1619d2DF63AF1c1eC2D7E413"  # Replace with your deployed contract address
        self.contract_abi = abi  # Replace with your contract's ABI
        self.pending_blockchain_updates = {}

        self.contract = self.w3.eth.contract(
            address=self.contract_address, abi=self.contract_abi
        )

        threading.Thread(target=self.check_and_aggregate_models).start()

    def updateBlockchain(self, aggregated_data):
        drone_ids = []
        performances = []
        reputations = []

        for drone_id, data in aggregated_data.items():
            drone_ids.append(int(drone_id))
            performances.append(int(data["performance"] * 10000))
            reputations.append(int(data["reputation"] * 10000))

        self.contract.functions.updateMultiplePerformanceAndReputation(
            drone_ids, performances, reputations
        ).transact({"from": self.w3.eth.accounts[0]})

    def queryBlockchain(self, drone_id):
        # Query model performance and reputation history
        (
            reputations,
            performances,
        ) = self.contract.functions.getAllReputationAndPerformance(int(drone_id)).call()

        # Convert to decimal
        performances_decimal = [p / 10000 for p in performances]
        reputations_decimal = [r / 10000 for r in reputations]

        # # Print all performances and reputations in decimal along with drone_id
        # for i in range(len(performances_decimal)):
        #     print(
        #         f"Drone ID: {drone_id}, Performance: {performances_decimal[i]}, Reputation: {reputations_decimal[i]}"
        #     )

        return performances_decimal, reputations_decimal

    def fed_evaluate(self, model, data_test_device):
        model.eval()
        with torch.no_grad():  # Do not calculate gradients to save memory
            outputs_test = model(data_test_device)

            predictions_test = self.to_predictions(outputs_test)

            # Calculate metrics
            # Calculate metrics
            accuracy = round(
                accuracy_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            precision = round(
                precision_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            recall = round(
                recall_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            f1 = round(f1_score(data_test_device.y.cpu(), predictions_test.cpu()), 4)
            return accuracy, precision, recall, f1

    def compute_loss(self, outputs, labels):
        # if self.global_model.num_output_features == 1:
        #     return criterion(outputs, labels)
        # elif self.global_model.num_output_features == 2:
        #     return criterion(outputs, labels.squeeze().long())
        # else:
        #     raise ValueError(
        #         "Invalid number of output features: {}".format(
        #             self.global_model.num_output_features
        #         )
        #     )
        return criterion(outputs, labels.squeeze().long())

    def to_predictions(self, outputs):
        # if self.global_model.num_output_features == 1:
        #     return (torch.sigmoid(outputs) > 0.5).float()
        # elif self.global_model.num_output_features == 2:
        #     return outputs.argmax(dim=1)
        # else:
        #     raise ValueError(
        #         "Invalid number of output features: {}".format(
        #             self.global_model.num_output_features
        #         )
        #     )
        return outputs.argmax(dim=1)

    def check_and_aggregate_models(self, use_reputation=True):
        torch.manual_seed(0)
        print("LOGGER-INFO: check_and_aggregate_models() is called")
        start_time = time.time()
        all_individual_accuracies = []
        aggregation_times = []
        # 检查队列中是否有足够的模型进行聚合·
        while True:
            self.new_model_event.wait()
            if self.local_models.qsize() >= 8:
                # 在这里更新区块链
                self.updateBlockchain(self.pending_blockchain_updates)
                self.pending_blockchain_updates.clear()  # 清空待更新的列表

                models_to_aggregate = []
                self.lock.acquire()
                try:
                    if use_reputation:
                        # 获取所有模型和它们的声誉
                        all_models = [self.local_models.get() for _ in range(8)]
                        all_reputations = [
                            self.reputation[drone_id]
                            for model_dict in all_models
                            for drone_id in model_dict
                        ]
                        # 根据声誉排序模型
                        sorted_indices = sorted(
                            range(len(all_reputations)),
                            key=lambda i: all_reputations[i],
                            reverse=True,
                        )
                        # 选择声誉最高的3个模型
                        models_to_aggregate = [
                            all_models[i] for i in sorted_indices[:8]
                        ]

                        # 打印每轮的所有节点声誉
                        print("                   ")
                        print(f"Current reputations: {self.reputation}")

                        # 打印参与聚合的节点
                        aggregated_node_ids = [
                            list(model_dict.keys())[0]
                            for model_dict in models_to_aggregate
                        ]
                        print(
                            f"Nodes participating in aggregation: {aggregated_node_ids}"
                        )

                    else:
                        # 如果不使用声誉，那么就选择所有的模型
                        for _ in range(8):
                            model_dict = self.local_models.get()
                            models_to_aggregate.append(model_dict)
                finally:
                    self.lock.release()

                individual_accuracies = []
                for model_dict in models_to_aggregate:
                    for drone_id, model in model_dict.items():
                        accuracy, precision, recall, f1 = self.fed_evaluate(
                            model, data_test_device
                        )
                        individual_accuracies.append(accuracy)
                #         print(f"Accuracy of model from drone {drone_id}: {accuracy}")

                # print("Individual accuracies: ", individual_accuracies)

                all_individual_accuracies.append(individual_accuracies)

                self.lock.acquire()  # Acquire lock before aggregation
                try:
                    self.aggregate_models(
                        models_to_aggregate, use_reputation=use_reputation
                    )
                finally:
                    self.lock.release()

                aggregation_times.append(time.time())

                accuracy, precision, recall, f1 = self.fed_evaluate(
                    self.aggregated_global_model, data_test_device
                )

                print(f"Aggregated model accuracy after aggregation: {accuracy}")

                self.aggregation_accuracies.append(accuracy)
                self.num_aggregations += 1  # 增加模型聚合的次数
                print("Aggregation accuracies so far: ", self.aggregation_accuracies)

                for drone_id, ip in self.drone_nodes.items():
                    self.send_model_thread(ip, "aggregated_global_model")

            # 当有n条记录就开始动态画图
            if self.num_aggregations == 30:
                end_time = time.time()  # 记录结束时间
                print(
                    f"Total time for aggregation: {end_time - start_time} seconds"
                )  # 打印执行时间

                # Find the time of the aggregation with the highest accuracy
                max_accuracy_index = self.aggregation_accuracies.index(
                    max(self.aggregation_accuracies)
                )
                max_accuracy_time = aggregation_times[max_accuracy_index]
                print(
                    f"Time of the aggregation with the highest accuracy: {max_accuracy_time - start_time} seconds"
                )

                plot_accuracy_vs_epoch(
                    self.aggregation_accuracies,
                    all_individual_accuracies,
                    self.num_aggregations,
                    learning_rate=0.02,
                )

                print("Program is about to terminate")

                for drone_id in self.drone_nodes:
                    print(
                        f"Debug: Current drone_id is {drone_id}, type is {type(drone_id)}"
                    )
                    try:
                        int_drone_id = int(drone_id)
                    except ValueError:
                        print(f"Error: Cannot convert drone_id {drone_id} to integer.")
                        continue  # Skip this iteration

                    reputation, performance = self.queryBlockchain(int_drone_id)
                    print(
                        f"Drone ID: {int_drone_id}, Performance: {performance}, Reputation: {reputation}"
                    )

                sys.exit()  # Terminate the program
            self.new_model_event.clear()

    def initialize_global_model(self):
        num_epochs = 15

        model = Net18(num_output_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        # Convert the model's output probabilities to binary predictions
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

        self.global_model = model

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation

    def aggregate_models(self, models_to_aggregate, use_reputation):
        if use_reputation:
            aggregated_model = weighted_average_aggregation(
                models_to_aggregate, self.reputation
            )
        else:
            aggregated_model = average_aggregation(models_to_aggregate)

        self.aggregated_global_model = Net18(num_output_features).to(device)  # 创建新的模型实例
        self.aggregated_global_model.load_state_dict(aggregated_model)  # 加载聚合后的权重

        # 保存全局模型到文件， 以pt形式保存
        torch.save(
            self.aggregated_global_model.state_dict(), "aggregated_global_model.pt"
        )

    def send_model(self, ip, model_type="global_model"):
        # 根据 model_type 加载不同的模型
        if model_type == "aggregated_global_model":
            model_path = "aggregated_global_model.pt"
            buffer = io.BytesIO()
            torch.save(self.aggregated_global_model.state_dict(), buffer)  # 修改这里

            # Get the contents of the buffer
            global_model_serialized = buffer.getvalue()

            # Encode the bytes as a base64 string
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()
            # 发送模型到子节点服务器
            # print("发送全局模型-发送！--->" + ip)
            response = requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            # print("发送全局模型-成功！--->" + ip)
            return json.dumps({"status": "success"})
        else:  # 默认加载 global_model
            model_path = "global_model.pt"
            # 检查模型是否已经存在
            if os.path.isfile(model_path):
                # 加载模型
                self.global_model = Net18(num_output_features).to(device)
                self.global_model.load_state_dict(torch.load(model_path))
                print(f"本地存在 {model_type}，发送中…… ")
            else:
                # 训练新模型
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
            # print("发送全局模型-发送！--->" + ip)
            response = requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            # print("发送全局模型-成功！--->" + ip)
            return json.dumps({"status": "success"})

    def compute_reputation(
        self, drone_id, performance, data_age, performance_threshold
    ):
        performance_contribution = sigmoid(performance)
        data_age_contribution = exponential_decay(data_age)
        reputation = performance_contribution * 0.9 + data_age_contribution * 0.1

        # 检查性能是否低于阈值，并更新连续低性能计数
        performance_threshold = 0.66
        # 检查并更新连续低性能计数
        if performance < performance_threshold:  # 选择合适的阈值
            if drone_id in self.low_performance_counts:
                self.low_performance_counts[drone_id] += 1

            else:
                self.low_performance_counts[drone_id] = 1
            print("low performance node,", drone_id)
            reputation = reputation * 0.1
        else:
            penalty_factor = 1 / (
                1 + np.exp(self.low_performance_counts.get(drone_id, 0))
            )

            if (
                drone_id in self.low_performance_counts
                and self.low_performance_counts[drone_id] > 0
            ):
                self.low_performance_counts[drone_id] -= 1
            print("use penalty factor")
            print(drone_id, " 看看有没有被惩罚 ", reputation)
            reputation = reputation * penalty_factor

        return reputation

    def send_model_thread(self, ip, model_type="aggregated_global_model"):
        threading.Thread(target=self.send_model, args=(ip, model_type)).start()

    # "0.0.0.0"表示应用程序在所有可用网络接口上运行
    def run(self, port=5000):
        app = Flask(__name__)
        print(criterion)

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
            # self.initialize_global_model()
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

            # 更新数据时效性标签
            if drone_id in self.data_age:
                self.data_age[drone_id] += 1
            else:
                self.data_age[drone_id] = 1

            reputation = self.compute_reputation(
                drone_id,
                performance,
                self.data_age[drone_id],
                performance_threshold=0.7,
            )

            self.update_reputation(drone_id, reputation)

            self.pending_blockchain_updates[drone_id] = {
                "performance": performance,
                "reputation": reputation,
            }

            self.local_models.put({drone_id: local_model})

            self.new_model_event.set()

            return jsonify({"status": "success"})

        app.run(host="localhost", port=port)


central_server_instance = CentralServer()
central_server_instance.run()
