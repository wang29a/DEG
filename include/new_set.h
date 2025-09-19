#ifndef NEW_SET_H
#define NEW_SET_H

#include <cstdint>
#include <list>
#include <vector>
#include "skyline_tree.h"

namespace stkq {
struct NEWNeighbor {
    unsigned id_;
    float emb_distance_;
    float geo_distance_;
    bool flag_;
    int layer_;
    std::list<int8_t> min_y_list;
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
        int insert_node_layer = 0;
        std::vector<std::list<float>> update_nodes;
        update_nodes.reserve(5);

        if (insert_pos != pool_.begin()) {
            auto prev_it = insert_pos - 1;

            int layer = 0;
            bool update = false;
            for (auto &min_y : prev_it->min_y_list) {
                if (e_d < min_y && !update) {
                    new_neighbor.min_y_list.push_back(e_d);
                    insert_node_layer = layer;
                    update = true;
                } else {
                    new_neighbor.min_y_list.push_back(min_y);
                }
                layer ++;
            }
            if (!update) {
                new_neighbor.min_y_list.push_back(e_d);
                insert_node_layer = layer;
                SKY_ASSERT(layer+1 == new_neighbor.min_y_list.size(), "新的一层");
            }
        } else {
            new_neighbor.min_y_list.push_back(e_d);
        }

        update_nodes.emplace_back(std::list<float>());
        update_nodes.at(0).push_back(e_d);

        for (auto it = insert_pos; it != pool_.end(); it ++) {
            // 更新min_y
            // bool update = false;
            // int layer = 0;
            // for (auto &min_y : it->min_y_list) {
            //     if (layer == insert_node_layer && e_d < min_y && !update) {
            //         update = true;
            //         min_y = e_d;
            //     }
            //     layer ++;
            // }
            // 受影响
            int layer = it->layer_ - insert_node_layer;
            if (layer < update_nodes.size()) {
                for (auto &e_distance : update_nodes.at(layer)) {
                    if (e_distance < it->emb_distance_) {
                        it->layer_ ++;
                        layer = it->layer_ - insert_node_layer;
                        if ((layer) < update_nodes.size()) {
                            update_nodes.at((layer)).emplace_back(it->emb_distance_);
                        } else {
                            update_nodes.push_back(std::list<float>());
                            update_nodes.at((layer)).emplace_back(it->emb_distance_);
                        }
                        break;
                    }
                }
            }
        }

        new_neighbor.layer_ = insert_node_layer;
        pool_.insert(insert_pos, new_neighbor);
        
        if (pool_.size() > M_) {
            int max_layer = 0;
            size_t idx = 0;
            for (size_t i = 0; i < pool_.size(); i ++) {
                auto &it = pool_.at(i);
                if (it.layer_ > max_layer) {
                    max_layer = it.layer_;
                    idx = i;
                }
            }
            pool_.erase(pool_.begin()+idx);
        }

        return ;
    }

    std::shared_ptr<std::vector<NEWNeighbor>> traverse() {
        std::vector<std::shared_ptr<std::vector<NEWNeighbor>>> layer_neighbor(20);
        for (auto &p: pool_) {
            if (layer_neighbor.at(p.layer_) == nullptr) {
                layer_neighbor.at(p.layer_) = std::make_shared<std::vector<NEWNeighbor>>();
                layer_neighbor.at(p.layer_)->reserve(20);
            }
            layer_neighbor.at(p.layer_)->emplace_back(p);
        }

        std::vector<NEWNeighbor> res;
        res.reserve(200);
        for (auto &layer : layer_neighbor) {
            if (layer == nullptr) {
                continue;
            }
            std::sort(layer->begin(), layer->end());
            for (auto &nei:*layer) {
                res.emplace_back(nei);
            }
        }

        return std::make_shared<std::vector<NEWNeighbor>>(res);
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