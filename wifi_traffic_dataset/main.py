from Central_Server import CentralServer
from Drone_Node import DroneNode
import multiprocessing
import time
import requests
from Dataset import features, labels

def create_drone_nodes(num_nodes):
    drone_nodes = []
    for i in range(num_nodes):
        drone_id = f"drone_{i}"
        ##此处存在问题
        local_data = features  # 为每个无人机节点提供本地数据
        drone_node = DroneNode(drone_id, local_data)
        drone_nodes.append(drone_node)
    return drone_nodes

def train_and_upload(drone_node, central_server_ip):
    drone_node.train_local_model()
    drone_node.upload_local_model(central_server_ip)


def wait_for_server(central_server_ip, port):
    url = f"http://{central_server_ip}:{port}/health_check"
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

if __name__ == '__main__':
    central_server = CentralServer()
    central_server.initialize_global_model()
    central_server_ip = "localhost"
    
    drone_nodes = create_drone_nodes(3)  # 根据您的需求创建适量的无人机节点
    central_server.distribute_global_model(drone_nodes)

    #创建一个新进程来运行中心服务器的Flask应用程序
    server_process = multiprocessing.Process(target=central_server.run)
    server_process.start()
    #central_server.run()  # 启动中心服务器的Flask应用程序

    # 确保服务器已启动,当调用这个函数的同时，确保输入正确的服务器地址和端口号
    wait_for_server("localhost", 5000)


    # 使用多进程模拟无人机节点的并行训练
    processes = []
    for drone_node in drone_nodes:
        process = multiprocessing.Process(target=train_and_upload, args=(drone_node, "localhost"))
        process.start()
        processes.append(process)

    # 等待所有无人机节点的训练和模型上传完成
    for process in processes:
        process.join()

    # 结束服务器进程
    server_process.terminate()


  