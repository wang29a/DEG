#ifndef CANDIDATE_TREE_H
#define CANDIDATE_TREE_H


#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <vector>

namespace stkq
{
    template<typename NodeId = int, typename NodeData = std::string>
    class DirectedGraph {
    public:
        struct Node {
            NodeId id;
            NodeData data;
            
            Node() = default;
            Node(const NodeId& node_id, const NodeData& node_data) 
                : id(node_id), data(node_data) {}
            Node(const NodeId& node_id, NodeData&& node_data) 
                : id(node_id), data(std::move(node_data)) {}
        };
    private:
        // 使用 unordered_set 提供 O(1) 平均时间复杂度的插入、删除和查找
        using AdjacencyList = std::unordered_set<NodeId>;
        using GraphContainer = std::unordered_map<NodeId, AdjacencyList>;
        
        GraphContainer adjacency_list_;
        std::unordered_map<NodeId, Node> nodes_;  // 存储节点ID到节点对象的映射
        
        // 为树结构优化：维护父子关系映射
        std::unordered_map<NodeId, NodeId> parent_map_;
        std::unordered_map<NodeId, std::unordered_set<NodeId>> children_map_;
        
    public:
        DirectedGraph() = default;
        ~DirectedGraph() = default;
        
        // 移动语义支持，提高性能
        DirectedGraph(DirectedGraph&&) = default;
        DirectedGraph& operator=(DirectedGraph&&) = default;
        
        // 禁用拷贝构造，避免意外的深拷贝开销
        DirectedGraph(const DirectedGraph&) = delete;
        DirectedGraph& operator=(const DirectedGraph&) = delete;
        
        // =============== 节点操作 ===============
        
        /**
        * 插入节点（带数据）- O(1) 平均时间复杂度
        */
        bool insertNode(const NodeId& node_id, const NodeData& data) {
            if (nodes_.find(node_id) != nodes_.end()) {
                return false; // 节点已存在
            }
            
            nodes_.emplace(node_id, Node{node_id, data});
            adjacency_list_[node_id]; // 创建空的邻接表
            children_map_[node_id];   // 创建空的子节点集合
            return true;
        }
        
        /**
        * 插入节点（移动语义）- O(1) 平均时间复杂度
        */
        bool insertNode(const NodeId& node_id, NodeData&& data) {
            if (nodes_.find(node_id) != nodes_.end()) {
                return false; // 节点已存在
            }
            
            nodes_.emplace(node_id, Node{node_id, std::move(data)});
            adjacency_list_[node_id]; // 创建空的邻接表
            children_map_[node_id];   // 创建空的子节点集合
            return true;
        }
        
        /**
        * 检查节点是否存在 - O(1) 平均时间复杂度
        */
        bool hasNode(const NodeId& node_id) const {
            return nodes_.find(node_id) != nodes_.end();
        }
        
        // =============== 边操作 ===============
        
        /**
        * 添加边 - O(1) 平均时间复杂度
        */
        bool addEdge(const NodeId& from, const NodeId& to) {
            // 确保两个节点都存在
            if (!hasNode(from) || !hasNode(to)) {
                return false;
            }
            
            // 避免自环（可选，根据需求调整）
            if (from == to) {
                return false;
            }
            
            bool inserted = adjacency_list_[from].insert(to).second;
            
            // 更新树结构信息（假设这是一个有向树）
            if (inserted) {
                parent_map_[to] = from;
                children_map_[from].insert(to);
            }
            
            return inserted;
        }
        
        /**
        * 删除边 - O(1) 平均时间复杂度
        */
        bool removeEdge(const NodeId& from, const NodeId& to) {
            auto from_it = adjacency_list_.find(from);
            if (from_it == adjacency_list_.end()) {
                return false;
            }
            
            bool removed = from_it->second.erase(to) > 0;
            
            // 更新树结构信息
            if (removed) {
                parent_map_.erase(to);
                children_map_[from].erase(to);
            }
            
            return removed;
        }
        
        /**
        * 检查边是否存在 - O(1) 平均时间复杂度
        */
        bool hasEdge(const NodeId& from, const NodeId& to) const {
            auto from_it = adjacency_list_.find(from);
            if (from_it == adjacency_list_.end()) {
                return false;
            }
            return from_it->second.find(to) != from_it->second.end();
        }
        
        // =============== 查询操作 ===============
        
        /**
        * 获取节点的所有邻居（出边指向的节点）- O(1) 访问邻接表
        * 返回 const 引用避免拷贝
        */
        const AdjacencyList& getNeighbors(const NodeId& node_id) const {
            static const AdjacencyList empty_set{};
            auto it = adjacency_list_.find(node_id);
            return (it != adjacency_list_.end()) ? it->second : empty_set;
        }
        std::vector<Node> getNeighborNodes(const NodeId& node_id) const {
            std::vector<Node> neighbor_nodes;
            const auto& neighbors = getNeighbors(node_id);
            neighbor_nodes.reserve(neighbors.size());
            
            for (const auto& neighbor_id : neighbors) {
                auto it = nodes_.find(neighbor_id);
                if (it != nodes_.end()) {
                    neighbor_nodes.push_back(it->second);
                }
            }
            return neighbor_nodes;
        }
        
        /**
        * 获取节点的所有子节点（树结构专用）- O(1) 访问
        */
        const std::unordered_set<NodeId>& getChildren(const NodeId& node_id) const {
            static const std::unordered_set<NodeId> empty_set{};
            auto it = children_map_.find(node_id);
            return (it != children_map_.end()) ? it->second : empty_set;
        }
        
        /**
        * 获取节点的父节点（树结构专用）- O(1) 访问
        */
        std::optional<NodeId> getParent(const NodeId& node_id) const {
            auto it = parent_map_.find(node_id);
            return (it != parent_map_.end()) ? std::optional<NodeId>{it->second} : std::nullopt;
        }
        
        /**
        * 获取节点的入度 - O(V) 时间复杂度（可优化为 O(1) 通过维护入度计数）
        */
        size_t getInDegree(const NodeId& node_id) const {
            if (!hasNode(node_id)) return 0;
            
            size_t in_degree = 0;
            for (const auto& [from_node, adj_set] : adjacency_list_) {
                if (adj_set.find(node_id) != adj_set.end()) {
                    ++in_degree;
                }
            }
            return in_degree;
        }
        
        /**
        * 获取节点的出度 - O(1) 时间复杂度
        */
        size_t getOutDegree(const NodeId& node_id) const {
            auto it = adjacency_list_.find(node_id);
            return (it != adjacency_list_.end()) ? it->second.size() : 0;
        }
        
        /**
        * 获取所有节点 - O(1) 访问（返回引用）
        */
        const std::unordered_set<NodeId>& getNodes() const {
            return nodes_;
        }
        
        // =============== 统计信息 ===============
        
        size_t getNodeCount() const { return nodes_.size(); }
        
        size_t getEdgeCount() const {
            size_t count = 0;
            for (const auto& [node, adj_set] : adjacency_list_) {
                count += adj_set.size();
            }
            return count;
        }
        
        bool isEmpty() const { return nodes_.empty(); }
    };

    struct NodeInfo {
        float emb_distance_;    // 嵌入距离
        float geo_distance_;    // 地理距离
        int layer_;            // 层级信息
    };
}
#endif