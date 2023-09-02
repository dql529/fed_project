from web3 import Web3
import json

with open("blockchain/build/contracts/ModelRegistry.json", "r", encoding="utf-8") as f:
    data = json.load(f)
abi = data["abi"]
# 初始化Web3
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))  # Ganache默认地址

# 合约地址和ABI
contract_address = "0xB00b340f5a6a3cb0bb5EF832fc3d6a04408316B9"
contract_abi = abi  # 你的合约ABI

# 创建合约对象
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# 设置发送交易的账户
account = "0x85606cc1d8390b31CA587510DFb0524699FD62f4"
private_key = "0x863202e83bb03578859d674f12d3109c2fbff5eeb45c4c8c016dd98102b26633"

# 准备交易数据
transaction = contract.functions.updatePerformanceAndReputation(
    1, 100, 100
).build_transaction(
    {
        "chainId": 1337,  # 适当的链ID
        "gas": 2000000,
        "gasPrice": w3.to_wei("20", "gwei"),
        "nonce": w3.eth.get_transaction_count(account),
    }
)

# 签名交易
signed_txn = w3.eth.account.sign_transaction(transaction, private_key)

# 发送交易
txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

# 等待交易被挖出
txn_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)

print("Transaction successful with receipt:", txn_receipt)
