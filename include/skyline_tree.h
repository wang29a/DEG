#ifndef SKYLINE_TREE_H
#define SKYLINE_TREE_H


#include "tree.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>

namespace stkq {
#define SKY_ASSERT(expr, message) assert((expr) && (message))

struct Record {
    unsigned key_;
    float emb_distance_;
    float geo_distance_;
    bool f_;
    Record() = default;
    Record(unsigned key, float emb_dist, float geo_dist, bool f) 
        : key_(key), emb_distance_(emb_dist), geo_distance_(geo_dist), f_(f) {}
};
struct RecordL {
    uint8_t layer_;
    unsigned key_;
    float emb_distance_;
    float geo_distance_;
    RecordL() = default;
    RecordL(unsigned key, float emb_dist, float geo_dist, uint8_t layer) 
        : layer_(layer), key_(key), emb_distance_(emb_dist), geo_distance_(geo_dist) {}
};

namespace skylinetree {

    class Node {
    public:
        using NodeId = unsigned;
        using NodePtr = std::shared_ptr<Node>;
        using ChildrenSet = std::set<NodePtr>;
        using ParentSet = std::set<NodePtr>;
    private:
        Record record_;
        ChildrenSet children_;
        ParentSet parents_;

    public:
        explicit Node(Record record) : record_(record) {
            children_.clear();
            parents_.clear();
        }

        // 拷贝构造函数
        Node(const Node& other) : record_(other.record_), 
                                  children_(other.children_), 
                                  parents_(other.parents_) {}

        // 移动构造函数
        Node(Node&& other) noexcept : record_(std::move(other.record_)),
                                      children_(std::move(other.children_)),
                                      parents_(std::move(other.parents_)) {}

        // 赋值操作
        Node& operator=(const Node& other) {
            if (this != &other) {
                record_ = other.record_;
                children_ = other.children_;
                parents_ = other.parents_;
            }
            return *this;
        }

        Node& operator=(Node&& other) noexcept {
            if (this != &other) {
                record_ = std::move(other.record_);
                children_ = std::move(other.children_);
                parents_ = std::move(other.parents_);
            }
            return *this;
        }

        // 比较运算
        auto operator<(const Node& other) const noexcept -> bool {
            return (record_.geo_distance_ < other.record_.geo_distance_ || 
                    (record_.geo_distance_ == other.record_.geo_distance_ && 
                    record_.emb_distance_ < other.record_.emb_distance_));
        }
        
        auto operator==(const Node& other) const noexcept -> bool {
            return record_.key_ == other.record_.key_;
        }
        
        auto operator!=(const Node& other) const noexcept -> bool {
            return !(*this == other);
        }

        auto getRecord() const -> const Record& { return record_; }

        Record& getRecord() { return record_; }
        
        const ChildrenSet& getChildren() const { return children_; }

        void addChild(NodePtr child) {
            children_.insert(std::move(child));
        }

        bool removeChild(NodePtr child) {
            return children_.erase(child) > 0;
        }

        const ParentSet& getParents() const { return parents_; }

        void addParent(NodePtr parent) {
            parents_.insert(std::move(parent));
        }

        void clearParents() {
            parents_.clear();
        }

        bool removeParent(NodePtr parent) {
            return parents_.erase(parent) > 0;
        }

    };
    class SkylineTree {
    public:
        using NodePtr = std::shared_ptr<Node>;

    private:
        std::vector<NodePtr> roots_;
        std::unordered_map<unsigned, NodePtr> node_index_;
        size_t M_;
    public:
        
        SkylineTree(size_t M) : M_(M) {
            roots_.reserve(8);
        }

        void constructe(std::vector<RecordL> &pools) {
            size_t layer0_count = 0;
            for (const auto& p : pools) {
                if (p.layer_ == 0) ++layer0_count;
            }
            roots_.reserve(layer0_count);
            for (auto &p : pools) {
                // todo
                if (p.layer_ == 0) {
                    Record record{p.key_, p.emb_distance_, p.geo_distance_, true};
                    auto node = std::make_shared<Node>(record);
                    roots_.push_back(node);
                    node_index_[p.key_] = node;
                } else {
                    insert(p.key_, p.emb_distance_, p.geo_distance_);
                }
            }
        }
        
        bool insert(unsigned id, float emb_distance, float geo_distance) {
            // 检查是否已存在
            if (node_index_.find(id) != node_index_.end()) {
                return false;  // 已存在
            }

            std::unordered_set<NodePtr> cur_layer;
            std::unordered_set<NodePtr> next_layer{roots_.begin(), roots_.end()};
            std::unordered_set<NodePtr> prev_layer_dominatingSet;
            std::unordered_set<NodePtr> prev_layer_dominatedSet;
            std::unordered_set<NodePtr> cur_layer_dominatingSet;
            std::unordered_set<NodePtr> cur_layer_dominatedSet;
            std::unordered_set<NodePtr> next_layer_dominatingSet;
            std::unordered_set<NodePtr> next_layer_dominatedSet;

            size_t cnt = 0;
            size_t layer = 0;
            cur_layer_dominatingSet.clear();

            do {
            // 没有被当前层点支配
                layer ++;
                // 在一层中若被支配，不可能支配同一层的其他点 只限有序层
                // SKY_ASSERT((cur_layer_dominatedSet.empty()), "在一层中若被支配，不可能支配同一层的其他点");
                prev_layer_dominatingSet.swap(cur_layer_dominatingSet);
                prev_layer_dominatedSet.merge(cur_layer_dominatedSet);
                cur_layer_dominatingSet.clear();
                cur_layer_dominatedSet.clear();
                cur_layer.swap(next_layer);
                next_layer.clear();

                for(auto& n: cur_layer) {
                    cnt ++;
                    auto &record = n->getRecord();
                    if (geo_distance > record.geo_distance_ && emb_distance > record.emb_distance_) {
                    // 支配插入点
                        cur_layer_dominatingSet.insert(n);
                    } else if (geo_distance == record.geo_distance_ && emb_distance == record.emb_distance_) {
                        std::cout<<"same distance"<< std::endl;
                        std::cout<< "  " << record.key_ << ", " << id << std::endl;
                        cur_layer_dominatedSet.insert(n);
                    } else if (geo_distance < record.geo_distance_ && emb_distance < record.emb_distance_) {
                    // 被插入点支配
                        cur_layer_dominatedSet.insert(n);
                    }
                    for (auto& child : n->getChildren()) {
                        next_layer.insert(child);
                    }
                }
                if (cnt > M_) {
                    return false;
                }
            } while(!cur_layer_dominatingSet.empty());

            // 同一层中 被插入点支配的点
            for(auto& n: next_layer) {
                auto &record = n->getRecord();
                // 支配插入点
                if (geo_distance < record.geo_distance_ && emb_distance < record.emb_distance_) {
                    next_layer_dominatedSet.insert(n);
                } else if (geo_distance > record.geo_distance_ && emb_distance > record.emb_distance_){
                    std::cout<<"error next layer node"<< std::endl;
                }
            }

            // 找到位置和被支配点 
            Record t = {id, emb_distance, geo_distance, true};
            std::shared_ptr<Node> insert_node = std::make_shared<Node>(t);

            // 同一层清除被支配点的父节点
            for (auto& child: cur_layer_dominatedSet) {
                insert_node->addChild(child);
                for (auto &parent_of_child : child->getParents()) {
                    parent_of_child->removeChild(child);
                }
                child->clearParents();
                child->addParent(insert_node);
                auto root_it = std::find(roots_.begin(), roots_.end(), child);
                if (root_it != roots_.end()) {
                    roots_.erase(root_it);
                }
            }

            // 加入下一层被支配点的父节点
            for (auto& child: next_layer_dominatedSet) {
                insert_node->addChild(child);
                child->addParent(insert_node);
            }

            for (auto& parent : prev_layer_dominatingSet) {
                parent->addChild(insert_node);
                insert_node->addParent(parent);
            }

            // if (prev_layer_dominatingSet.empty()) {
            //     roots_.emplace_back(insert_node);
            // }

            if (layer == 1 && cur_layer_dominatingSet.empty()) {
                roots_.emplace_back(insert_node);
                std::sort(roots_.begin(), roots_.end());
            }

            node_index_[id] = insert_node;
            return true;
        }

        bool remove_with_index(unsigned id) {
            auto it = node_index_.find(id);
            if (it == node_index_.end()) {
                return false;  // 不存在
            }

            auto remove_node = it->second;
            const auto& children = remove_node->getChildren();
            const auto& parents = remove_node->getParents();

            // 重连父子关系
            for (auto& child : children) {
                child->removeParent(remove_node);
                for (auto& parent : parents) {
                    child->addParent(parent);
                    parent->addChild(child);
                }
            }

            // 从父节点中移除
            for (auto& parent : parents) {
                parent->removeChild(remove_node);
            }

            // 如果是根节点，从根节点列表中移除
            auto root_it = std::find(roots_.begin(), roots_.end(), remove_node);
            if (root_it != roots_.end()) {
                roots_.erase(root_it);
            }

            node_index_.erase(it);

            return true;
        }

        bool remove(unsigned id, float emb_distance, float geo_distance) {

            std::vector<NodePtr> cur_layer = roots_;
            std::vector<NodePtr> next_layer;
            NodePtr remove_node = nullptr;

            while (true) {
                next_layer.clear();

                for(auto& n: cur_layer) {
                    auto &record = n->getRecord();
                    if (geo_distance > record.geo_distance_ && emb_distance > record.emb_distance_) {
                    // 支配删除点
                        for (auto& children : n->getChildren()) {
                            next_layer.emplace_back(children);
                        }
                    } else if (geo_distance == record.geo_distance_ && emb_distance == record.emb_distance_) {
                        if (record.key_ == id) {
                            remove_node = n;
                            break;       
                        }
                    }
                }
                cur_layer.swap(next_layer);
            }

            if (remove_node == nullptr) {
                return false;
            }

            auto &childrens = remove_node->getChildren();
            auto &parents = remove_node->getParents();

            for (auto &child : childrens) {
                child->removeParent(remove_node);
                for (auto &parent : parents) {
                    child->addParent(parent);
                    parent->addChild(child);
                }
            }
            for (auto &parent : parents) {
                parent->removeChild(remove_node);
            }

            // 如果是根节点，从根节点列表中移除
            auto root_it = std::find(roots_.begin(), roots_.end(), remove_node);
            if (root_it != roots_.end()) {
                roots_.erase(root_it);
            }

            auto it = node_index_.find(id);
            node_index_.erase(it);

            return true;
        }

        std::shared_ptr<std::vector<RecordL>> traverse(size_t range) {
            std::shared_ptr<std::vector<RecordL>> res = std::make_shared<std::vector<RecordL>>();
            res->reserve(range);
            std::unordered_set<NodePtr> next_layer;
            uint8_t layer = 0;
            std::unordered_set<NodePtr> cur_layer{roots_.begin(), roots_.end()};
            std::sort(roots_.begin(), roots_.end(), [] (const NodePtr a, const NodePtr b) {
                // const auto &r_a = a->getRecord();
                // const auto &r_b = b->getRecord();
                return *a < *b;
            });
            std::set<unsigned> id_set;
            while (!cur_layer.empty() && res->size() < range) {
                next_layer.clear();

                for (auto &n : cur_layer) {
                    auto &record = n->getRecord();
                    if (id_set.find(record.key_) == id_set.end()) {
                        id_set.insert(record.key_);
                        res->emplace_back(record.key_, record.emb_distance_, record.geo_distance_, layer);
                    }
                    for (auto& children : n->getChildren()){
                        next_layer.insert(children);
                    }
                }

                cur_layer = std::move(next_layer);
                layer ++;
            }

            return res;
        }

        std::shared_ptr<std::vector<NodePtr>> traverse_layer() {
            std::vector<NodePtr> res;
            std::vector<NodePtr> prev_layer;
            std::vector<NodePtr> next_layer;
            std::vector<NodePtr> cur_layer;
            cur_layer = roots_;
            while(!cur_layer.empty()) {
                next_layer.clear();
                res.clear();

                for (auto &n : cur_layer) {
                    auto &record = n->getRecord();
                    if (record.f_) {
                        res.emplace_back(n);
                    }
                    for (auto& children : n->getChildren()){
                        next_layer.emplace_back(children);
                    }
                }
                if (!res.empty()) {
                    return std::make_shared<std::vector<NodePtr>>(res);
                }
                prev_layer = std::move(cur_layer);
                cur_layer = std::move(next_layer);
            }

            return nullptr;
        }

        const NodePtr getNode(unsigned id) {
            auto it = node_index_.find(id);
            if (it == node_index_.end()) {
                return nullptr;
            }
            return it->second;
        }

        size_t size() const noexcept { return node_index_.size(); }
        size_t rootCount() const noexcept { return roots_.size(); }
    };
}
}

#endif