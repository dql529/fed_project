// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract ModelRegistry {
    struct NodeInfo {
        uint256 reputation;
        uint256 performance;
    }

    mapping(uint256 => NodeInfo) public nodes;  // Mapping from droneId to NodeInfo
    mapping(uint256 => NodeInfo[]) public nodesHistory;  // Mapping from droneId to an array of NodeInfo

    // 更新性能和声誉，并将其添加到历史记录中
    function updatePerformanceAndReputation(uint256 droneId, uint256 _performance, uint256 _reputation) public {
        NodeInfo memory newNodeInfo = NodeInfo(_performance, _reputation);
        nodes[droneId] = newNodeInfo;  // 更新当前节点信息
        nodesHistory[droneId].push(newNodeInfo);  // 添加到历史记录
    }

    // 获取最新的性能和声誉
    function getLatestReputationAndPerformance(uint256 droneId) public view returns (uint256, uint256) {
        uint256 length = nodesHistory[droneId].length;
        if (length == 0) {
            return (0, 0);  // 返回默认值，如果没有历史记录
        }
        NodeInfo memory latestInfo = nodesHistory[droneId][length - 1];
        return (latestInfo.reputation, latestInfo.performance);
    }

    // 获取所有历史记录
    function getAllReputationAndPerformance(uint256 droneId) public view returns (uint256[] memory, uint256[] memory) {
        uint256 length = nodesHistory[droneId].length;
        uint256[] memory reputations = new uint256[](length);
        uint256[] memory performances = new uint256[](length);

        for (uint256 i = 0; i < length; i++) {
            NodeInfo memory info = nodesHistory[droneId][i];
            reputations[i] = info.reputation;
            performances[i] = info.performance;
        }

        return (reputations, performances);
    }
}
