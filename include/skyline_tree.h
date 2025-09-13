#ifndef SKYLINE_TREE_H
#define SKYLINE_TREE_H


#include "tree.h"
#include "index.h"
#include <memory>
#include <set>
#include <cassert>

namespace stkq {
#define SKY_ASSERT(expr, message) assert((expr) && (message))

struct Record {
    unsigned key_;
    float emb_distance_;
    float geo_distance_;
};

struct NodeInfo {
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
        ParentSet parent_;

    public:
        explicit Node(Record record) : record_(record) {
            children_.clear();
        }

        // 拷贝构造函数

        // 移动构造函数

        // 赋值操作

        // 比较运算
        auto operator<(const Node& other) const -> bool {
            return record_.key_ < other. record_.key_;
        }
        
        auto operator==(const Node& other) const -> bool {
            return record_.key_ == other.record_.key_;
        }
        
        auto operator!=(const Node& other) const -> bool {
            return !(*this == other);
        }

        auto getRecord() const -> const Record& { return record_; }

        Record& getRecord() { return record_; }
        
        const ChildrenSet& getChildren() const { return children_; }

        void addChildren(NodePtr child) {
            children_.insert(child);
        }

        bool removeChildren(NodePtr child) {
            return children_.erase(child) > 0;
        }

        void addParent(NodePtr parent) {
            parent_.insert(parent);
        }

        void clearParent() {
            parent_.clear();
        }

        bool removeParent(NodePtr parent) {
            return parent_.erase(parent) > 0;
        }

    };
    class SkylineTree {
    public:
        using NodePtr = std::shared_ptr<Node>;

    private:
        std::vector<NodePtr> roots;

    public:

        void constructe(std::vector<Index::DEGNNDescentNeighbor> &pools) {
            for (auto &p : pools) {
                // todo
                if (p.layer_ == 0) {
                    Record t = {p.id_, p.emb_distance_, p.geo_distance_};
                    roots.emplace_back(std::make_shared<Node>(std::move(t)));
                } else {
                    insert(p.id_, p.emb_distance_, p.geo_distance_, p.layer_);
                }
            }
        }
        
        bool insert(unsigned id, float emb_distance, float geo_distance, int layer) {

            std::vector<NodePtr> cur_layer;
            std::vector<NodePtr> next_layer;
            std::vector<NodePtr> prev_layer_dominatingSet;
            std::vector<NodePtr> prev_layer_dominatedSet;
            std::vector<NodePtr> cur_layer_dominatingSet;
            std::vector<NodePtr> cur_layer_dominatedSet;
            std::vector<NodePtr> next_layer_dominatingSet;
            std::vector<NodePtr> next_layer_dominatedSet;

            for (auto& root : roots) {
                // cur_layer.emplace_back(root);
                auto &record = root->getRecord();
                if (geo_distance >= record.geo_distance_ && emb_distance >= record.emb_distance_) {
                // 支配插入点
                    cur_layer_dominatingSet.emplace_back(root);
                } else if (geo_distance == record.geo_distance_ && emb_distance == record.emb_distance_) {
                } else if (geo_distance < record.geo_distance_ && emb_distance < record.emb_distance_) {
                // 被插入点支配
                    cur_layer_dominatedSet.emplace_back(root);
                }

                for (auto& children : root->getChildren()) {
                    bool is_parent = (children->getParent() == root);
                    SKY_ASSERT(is_parent, "");
                    next_layer.emplace_back(children);
                }
            }

            // 没有被当前层点支配
            while (cur_layer_dominatingSet.empty()) {
                // 在一层中若被支配，不可能支配同一层的其他点
                SKY_ASSERT((cur_layer_dominatedSet.empty()), "在一层中若被支配，不可能支配同一层的其他点");
                prev_layer_dominatingSet.swap(cur_layer_dominatingSet);
                prev_layer_dominatedSet.swap(cur_layer_dominatedSet);
                cur_layer_dominatingSet.clear();
                cur_layer_dominatedSet.clear();
                cur_layer.swap(next_layer);
                next_layer.clear();

                for(auto& n: cur_layer) {
                    auto &record = n->getRecord();
                    if (geo_distance >= record.geo_distance_ && emb_distance >= record.emb_distance_) {
                    // 支配插入点
                        cur_layer_dominatingSet.emplace_back(n);
                    } else if (geo_distance == record.geo_distance_ && emb_distance == record.emb_distance_) {
                    } else if (geo_distance < record.geo_distance_ && emb_distance < record.emb_distance_) {
                    // 被插入点支配
                        cur_layer_dominatedSet.emplace_back(n);
                    }
                    for (auto& children : n->getChildren()) {
                        next_layer.emplace_back(children);
                    }
                }
            }

            cur_layer.swap(next_layer);
            next_layer.clear();
            // 同一层中 被插入点支配的点
            for(auto& n: cur_layer) {
                auto &record = n->getRecord();
                // 支配插入点
                if (geo_distance >= record.geo_distance_ && emb_distance >= record.emb_distance_) {
                    next_layer_dominatingSet.emplace_back(n);
                } else if (geo_distance == record.geo_distance_ && emb_distance == record.emb_distance_) {
                // 被插入点支配
                } else if (geo_distance < record.geo_distance_ && emb_distance < record.emb_distance_) {
                    next_layer_dominatedSet.emplace_back(n);
                }
            }

            // 找到位置和被支配点 
            Record t = {id, emb_distance, geo_distance};
            std::shared_ptr<Node> insert_node = std::make_shared<Node>(t);

            // 同一层清除被支配点的父节点
            for (auto& child: cur_layer_dominatedSet) {
                insert_node->addChildren(child);
                child->clearParent();
                child->addParent(insert_node);
            }

            // 加入下一层被支配点的父节点
            for (auto& child: next_layer_dominatedSet) {
                insert_node->addChildren(child);
                child->addParent(insert_node);
            }

            for (auto& parent : prev_layer_dominatingSet) {
                parent->addChildren(insert_node);
            }

            return true;
        }

        bool remove(unsigned id) {

            return true;
        }

        // void 
    };
}
}

#endif