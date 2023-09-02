from web3 import Web3
import json

with open("blockchain/build/contracts/ModelRegistry.json", "r", encoding="utf-8") as f:
    data = json.load(f)
abi = data["abi"]
# 初始化Web3
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))  # Ganache默认地址

# 合约地址和ABI
contract_address = "0x1f35D688f4d033B433603f36ddB59Fa9bF5720Ab"
contract_abi = abi  # 你的合约ABI
# 初始化合约
contract = w3.eth.contract(address=contract_address, abi=contract_abi)


def updateBlockchain(self, drone_id, performance, reputation):
    drone_id_uint256 = int(drone_id)
    performance_uint256 = int(performance * 10000)  # 假设你想保留4位小数
    reputation_uint256 = int(reputation * 10000)  # 假设你想保留4位小数

    self.contract.functions.updatePerformanceAndReputation(
        drone_id_uint256, performance_uint256, reputation_uint256
    ).transact({"from": self.w3.eth.accounts[0]})


def queryBlockchain(self, drone_id):
    drone_id_uint256 = int(drone_id)
    performance, reputation = self.contract.functions.getReputationAndPerformance(
        drone_id_uint256
    ).call()
