#ifndef NEW_SET_H
#define NEW_SET_H

#include <cstdint>
#include <list>
#include <vector>
#include "skyline_tree.h"

namespace stkq {
struct NEWNeighbor {
    unsigned id_;           // 较少访问
    float geo_distance_;     // 用于比较的主要字段
    float emb_distance_;     // 用于比较的次要字段
    int layer_;             // 频繁访问
    bool flag_;            // 较少访问
    std::vector<float> min_y_list; // 大对象放最后
    NEWNeighbor() = default;
    NEWNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer)
        : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag_(f), layer_(layer)
    { }
    NEWNeighbor(unsigned id, float emb_distance, float geo_distance)
        : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance)
    { }
    NEWNeighbor(const NEWNeighbor &other)
        : id_{other.id_}, emb_distance_{other.emb_distance_}, geo_distance_(other.geo_distance_), flag_(other.flag_), layer_(other.layer_)
    { }
    inline bool operator<(const NEWNeighbor &other) const {
        return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
    }
};

struct NEWSimpleNeighbor {
    float emb_distance_;
    NEWSimpleNeighbor() = default;
    NEWSimpleNeighbor(unsigned id, float emb_distance, float geo_distance, int layer)
        : emb_distance_{emb_distance}
    { }
};

class NEWSkyLine {
private:
    std::vector<NEWNeighbor> pool_;
    unsigned M_; // 记录pool的最大大小, 也就是candidate边数
    std::vector<std::vector<float>> update_nodes_;
    
public:
    NEWSkyLine() = default;

    NEWSkyLine(unsigned l) : M_(l) {
        pool_.reserve(M_);
    }

    void insert(unsigned id, float e_d, float s_d) {
        // pool_.emplace_back(id, e_d, s_d, true, -1);
        // std::sort(pool_.begin(), pool_.end());
        NEWNeighbor new_neighbor{id, e_d, s_d, true, -1};
        auto insert_pos = std::lower_bound(pool_.begin(), pool_.end(), new_neighbor);
        size_t insert_idx = insert_pos - pool_.begin();
        int insert_node_layer = 0;
        update_nodes_.clear();
        update_nodes_.reserve(10);

        if (insert_idx > 0) {
            const auto& prev_neighbor = pool_[insert_idx - 1];

            new_neighbor.min_y_list.reserve(prev_neighbor.min_y_list.size() + 1);
            int layer = 0;
            bool update = false;
            for (size_t i = 0; i < prev_neighbor.min_y_list.size(); ++i) {
                float min_y = prev_neighbor.min_y_list[i];
                if (e_d < min_y && !update) {
                    new_neighbor.min_y_list.push_back(e_d);
                    insert_node_layer = i;
                    update = true;
                    new_neighbor.min_y_list.insert(
                        new_neighbor.min_y_list.end(),
                        prev_neighbor.min_y_list.begin() + i,
                        prev_neighbor.min_y_list.end()
                    );
                    break;
                } else {
                    new_neighbor.min_y_list.push_back(min_y);
                }
            }
            if (!update) {
                new_neighbor.min_y_list.push_back(e_d);
                insert_node_layer = layer;
                SKY_ASSERT(layer+1 == new_neighbor.min_y_list.size(), "新的一层");
            }
        } else {
            new_neighbor.min_y_list.push_back(e_d);
        }

        new_neighbor.layer_ = insert_node_layer;

        update_nodes_.emplace_back();
        update_nodes_[0].reserve(4);
        update_nodes_.at(0).push_back(e_d);

        for (size_t i = insert_idx; i < pool_.size(); ++i) {
            auto& current_neighbor = pool_[i];

            if (current_neighbor.layer_ < insert_node_layer) continue;
            // 受影响
            size_t layer_diff = current_neighbor.layer_ - insert_node_layer;
            if (layer_diff < update_nodes_.size()) {
                const auto& update_list = update_nodes_[layer_diff];
                // 优化10: 使用范围for循环，编译器更容易优化
                for (float e_distance : update_list) {
                    if (e_distance < current_neighbor.emb_distance_) {
                        current_neighbor.layer_++;
                        size_t new_layer_diff = current_neighbor.layer_ - insert_node_layer;
                        
                        // 确保update_nodes_有足够空间
                        while (update_nodes_.size() <= new_layer_diff) {
                            update_nodes_.emplace_back();
                            update_nodes_.back().reserve(4);
                        }
                        
                        update_nodes_[new_layer_diff].push_back(current_neighbor.emb_distance_);
                        break;
                    }
                }
            }
        }

        pool_.insert(pool_.begin() + insert_idx, std::move(new_neighbor));
        
        if (pool_.size() > M_) {
            int max_layer = -1;
            size_t remove_idx = 0;
            for (size_t i = pool_.size(); i > 0; --i) {
                size_t idx = i - 1;
                if (pool_[idx].layer_ > max_layer) {
                    max_layer = pool_[idx].layer_;
                    remove_idx = idx;
                }
            }
            pool_.erase(pool_.begin()+remove_idx);
        }

        return ;
    }

    std::vector<NEWNeighbor> traverse() {
        std::vector<NEWNeighbor> res;

        res.assign(pool_.begin(), pool_.end());
        std::sort(res.begin(), res.end(), 
              [](const auto& a, const auto& b) {
                  return a.layer_ < b.layer_ || 
                         (a.layer_ == b.layer_ && a < b);
              });

        return res;
    }

    std::vector<size_t> traverse_layer() {
        std::vector<size_t> res;
        res.reserve(20);
        int layer = -1;
        for (size_t i = 0; i < pool_.size(); i++) {
            auto &p = pool_.at(i);
            if (layer == -1 && p.flag_) {
                res.emplace_back(i);
                layer = p.layer_;
            } else if (p.layer_ == layer && p.flag_) {
                res.emplace_back(i);
            }
        }

        return res;
    }

    NEWNeighbor& at(size_t idx) {
        return pool_.at(idx);
    }

    const NEWNeighbor& at(size_t idx) const {
        return pool_.at(idx);
    }
 
    void remove() {
        return ;
    }
    size_t size() {
        return pool_.size();
    }

};
}

#endif