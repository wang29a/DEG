#ifndef STKQ_INDEX_H
#define STKQ_INDEX_H

#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <functional>
#include <map>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include "util.h"
#include "distance.h"
#include "parameters.h"
#include "policy.h"
#include "rtree.h"
#include "CommonDataStructure.h"
#include <mm_malloc.h>
#include <stdlib.h>
#define INF_N -std::numeric_limits<float>::max()
#define INF_P std::numeric_limits<float>::max()

namespace stkq
{
    typedef std::lock_guard<std::mutex> LockGuard;
    class NNDescent
    {
    public:
        unsigned K;
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;
        struct Neighbor
        {
            unsigned id;
            float distance;
            bool flag;

            Neighbor() = default;

            Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

            inline bool operator<(const Neighbor &other) const
            {
                return distance < other.distance;
            }

            inline bool operator>(const Neighbor &other) const
            {
                return distance > other.distance;
            }
        };

        struct nhood
        {
            std::mutex lock;            // 互斥锁 用于并发访问控制
            std::vector<Neighbor> pool; // 存储邻居节点的优先队列 最小堆
            unsigned M;                 // 记录pool的大小

            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;

            nhood() {}

            nhood(unsigned l, unsigned s)
            {
                M = s;
                nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                // resize用于定义向量的实际大小（元素数量） 而reserve用于优化内存分配以应对向量大小的潜在增长
                pool.reserve(l);
            }

            nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N)
            {
                M = s;
                nn_new.resize(s * 2);
                GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N); // 还用随机数填充 nn_new
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(const nhood &other)
            {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void insert(unsigned id, float dist)
            {
                // 方法用于向 pool 中插入新的邻居节点，如果新节点的距离小于 pool 中最远节点的距离，则替换之
                //  pool 最大堆
                LockGuard guard(lock);
                if (dist > pool.front().distance)
                    return;
                for (unsigned i = 0; i < pool.size(); i++)
                {
                    if (id == pool[i].id)
                        return;
                }
                if (pool.size() < pool.capacity())
                {
                    pool.push_back(Neighbor(id, dist, true));
                    std::push_heap(pool.begin(), pool.end());
                }
                else
                {
                    std::pop_heap(pool.begin(), pool.end());
                    // 把最大的放到最后一个
                    pool[pool.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(pool.begin(), pool.end());
                }
            }

            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    // 遍历 nn_new（新邻居）集合中的每个元素 i
                    for (unsigned const j : nn_new)
                    {
                        // 再次遍历 nn_new 集合中的每个元素 j
                        if (i < j)
                        {
                            callback(i, j);
                            // 如果 i 小于 j（避免重复处理和自连接），调用 callback 函数处理这对邻居节点
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        callback(i, j);
                        // 接着遍历 nn_old（旧邻居）集合中的每个元素 j，对每个新旧邻居对调用 callback 函数
                    }
                }
            }
            // 似乎是NN Descent
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
        {
            // find the location to insert
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance)
            {
                memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance)
            {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1)
            {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)
                    right = mid;
                else
                    left = mid;
            }
            // check equal ID

            while (left > 0)
            {
                if (addr[left].distance < nn.distance)
                    break;
                if (addr[left].id == nn.id)
                    return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)
                return K + 1;
            memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        typedef std::vector<nhood> KNNGraph;
        KNNGraph graph_;
    };

    class NSW
    {
    public:
        unsigned NN_;
        unsigned ef_construction_ = 150; // l
        unsigned n_threads_ = 32;
    };

    class HNSW
    {
    public:
        unsigned m_ = 12; // k
        unsigned max_m_ = 12;
        unsigned max_m0_ = 24;
        int mult;
        float level_mult_ = 1 / log(1.0 * m_);

        int max_level_ = 0;
        mutable std::mutex max_level_guard_;

        template <typename KeyType, typename DataType>
        class MinHeap
        { // 这是一个实现最小堆数据结构的类。最小堆是一种树形数据结构，用于高效地管理和检索元素，其中父节点的键（key）总是小于或等于其子节点的键
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key > i2.key;
                }
            };

            MinHeap()
            {
            }

            const KeyType top_key()
            {
                if (v_.size() <= 0)
                    return 0.0;
                return v_[0].key;
            }

            Item top()
            {
                if (v_.size() <= 0)
                    throw std::runtime_error("[Error] Called top() operation with empty heap");
                return v_[0];
            }
            // 返回堆顶元素 最小元素

            void pop()
            {
                std::pop_heap(v_.begin(), v_.end());
                v_.pop_back();
            }
            // 移除堆顶元素

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));
                std::push_heap(v_.begin(), v_.end());
            }

            // 向堆中添加新元素

            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class HnswNode
        {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level + 1)
            {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) { level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode *> &GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<HnswNode *> &new_friends)
            {
                if (level >= friends_at_layer_.size())
                    friends_at_layer_.resize(level + 1);
                friends_at_layer_[level].swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

            // 1. The list of friends is sorted
            // 2. bCheckForDup == true addFriend checks for duplicates using binary searching
            inline void AddFriends(HnswNode *element, bool bCheckForDup)
            {
                std::unique_lock<std::mutex> lock(access_guard_);
                if (bCheckForDup)
                {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if (it == friends_at_layer_[0].end() || (*it) != element)
                    {
                        friends_at_layer_[0].insert(it, element);
                    }
                }
                else
                {
                    friends_at_layer_[0].push_back(element);
                }
            }

        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;

            std::vector<std::vector<HnswNode *>> friends_at_layer_;
            std::mutex access_guard_;
        };

        class FurtherFirst
        {
        public:
            FurtherFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const FurtherFirst &n) const
            {
                return (distance_ < n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较低” 因此 FurtherFirst 实际上是用于构建最大堆的
        };

        class CloserFirst
        {
        public:
            CloserFirst(HnswNode *node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode *GetNode() const { return node_; }
            bool operator<(const CloserFirst &n) const
            {
                return (distance_ > n.GetDistance());
            }

        private:
            HnswNode *node_;
            float distance_;
            // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        };

        typedef typename std::pair<HnswNode *, float> IdDistancePair;
        struct IdDistancePairMinHeapComparer
        {
            bool operator()(const IdDistancePair &p1, const IdDistancePair &p2) const
            {
                return p1.second > p2.second;
            }
        };
        typedef typename boost::heap::d_ary_heap<IdDistancePair, boost::heap::arity<4>, boost::heap::compare<IdDistancePairMinHeapComparer>> IdDistancePairMinHeap;
        // IdDistancePairMinHeap 是一个 4-叉最小堆，使用 boost::heap::d_ary_heap 实现，并通过 IdDistancePairMinHeapComparer 来比较元素
        // 这样构造的堆会将最小的距离元素保持在顶部
        HnswNode *enterpoint_ = nullptr;
        std::vector<HnswNode *> nodes_;
    };

    class NSG
    {
    public:
        unsigned R_refine;
        unsigned L_refine;
        unsigned C_refine;

        unsigned ep_;
        unsigned width;
    };

    class SSG
    {
    public:
        float A;
        unsigned n_try;
        // unsigned width;

        std::vector<unsigned> eps_;
        unsigned test_min = INT_MAX;
        unsigned test_max = 0;
        long long test_sum = 0;
    };

    // class Mbr
    // {
    // public:
    //     static const unsigned DIM = 2;
    //     std::vector<std::vector<float>> coord;
    //     std::vector<unsigned> object_ids;
    //     Mbr()
    //     {
    //         init();
    //     };

    //     Mbr(float xmin, float xmax, float ymin, float ymax)
    //     {
    //         std::vector<float> x_coor = {xmin, xmax};
    //         std::vector<float> y_coor = {ymin, ymax};
    //         coord.emplace_back(x_coor);
    //         coord.emplace_back(y_coor);
    //     };

    //     void init()
    //     {
    //         std::vector<float> v = {INF_P, INF_N};
    //         for (size_t dim = 0; dim < DIM; dim++)
    //             coord.emplace_back(v);
    //     };
    //     float getArea() const
    //     {
    //         float area = 1;
    //         for (size_t dim = 0; dim < DIM; dim++)
    //             area *= coord[dim][1] - coord[dim][0];
    //         return area;
    //     };

    //     float getMargin() const
    //     {
    //         float margin = 0;
    //         for (size_t dim = 0; dim < DIM; dim++)
    //             margin += coord[dim][1] - coord[dim][0];
    //         return margin;
    //     };

    //     void enlarge(Mbr &add)
    //     {
    //         for (size_t dim = 0; dim < DIM; dim++)
    //         {
    //             coord[dim][0] = std::min(coord[dim][0], add.coord[dim][0]);
    //             coord[dim][1] = std::max(coord[dim][1], add.coord[dim][1]);
    //         }
    //     };

    //     inline float getCenter(size_t dim)
    //     {
    //         return (coord[dim][0] + coord[dim][1]) * 0.5;
    //     }

    //     inline bool isPoint()
    //     {
    //         for (size_t i = 0; i < DIM; i++)
    //         {
    //             if (coord[i][0] != coord[i][1])
    //                 return false;
    //         }
    //         return true;
    //     }

    //     bool operator==(const Mbr &m) const
    //     {
    //         return m.coord == coord;
    //     }

    //     static Mbr getMbr(Mbr &mbr1, Mbr &mbr2)
    //     {
    //         Mbr mbr;
    //         for (size_t dim = 0; dim < DIM; dim++)
    //         {
    //             mbr.coord[dim][0] = std::min(mbr1.coord[dim][0], mbr2.coord[dim][0]);
    //             mbr.coord[dim][1] = std::max(mbr1.coord[dim][1], mbr2.coord[dim][1]);
    //         }
    //         return mbr;
    //     };
    //     // get the overlap area of two MBRs
    //     static float getOverlap(Mbr &mbr1, Mbr &mbr2)
    //     {
    //         float overlap = 1;
    //         for (size_t dim = 0; dim < DIM; dim++)
    //         {
    //             float maxMin = std::max(mbr1.coord[dim][0], mbr2.coord[dim][0]);
    //             float minMax = std::min(mbr1.coord[dim][1], mbr2.coord[dim][1]);
    //             if (maxMin >= minMax)
    //                 return 0;
    //             overlap *= minMax - maxMin;
    //         }
    //         return overlap;
    //     };

    //     static bool isOverlap(Mbr &mbr1, Mbr &mbr2)
    //     {
    //         if (mbr2.isPoint())
    //         {
    //             for (size_t dim = 0; dim < DIM; dim++)
    //             {
    //                 if (mbr2.coord[dim][1] > mbr1.coord[dim][1] || mbr2.coord[dim][0] < mbr1.coord[dim][0])
    //                 {
    //                     return false;
    //                 }
    //             }
    //         }
    //         else
    //         {
    //             float dis = Mbr::getOverlap(mbr1, mbr2);
    //             if (dis == 0)
    //             {
    //                 return false;
    //             }
    //         }
    //         return true;
    //     };
    // };

    class GEOGRAPH
    {
    public:
        struct GeoGraphNeighbor
        {
            unsigned id_;
            float emb_distance_;
            float geo_distance_;
            unsigned layer_;
            std::vector<std::pair<float, float>> available_range;

            GeoGraphNeighbor() = default;
            GeoGraphNeighbor(unsigned id, float emb_distance, float geo_distance) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance)
            {
                available_range.emplace_back(0, 1);
            }
            GeoGraphNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range) {}
            GeoGraphNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range, unsigned l) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range), layer_(l) {}

            inline bool operator<(const GeoGraphNeighbor &other) const
            {
                // return geo_distance_ < other.geo_distance_;
                // 较小的 geo_distance_ 值会被排序到较前的位置
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
            }
        };

        struct GeoGraphSimpleNeighbor
        {
            unsigned id_;
            // unsigned layer_;
            std::vector<std::pair<float, float>> available_range;

            GeoGraphSimpleNeighbor() = default;
            GeoGraphSimpleNeighbor(unsigned id, std::vector<std::pair<float, float>> range) : id_{id}, available_range(range) {}
        };

        struct GeoGraphNNDescentNeighbor
        {
            unsigned id_;
            float emb_distance_;
            float geo_distance_;
            bool flag;
            int layer_;
            GeoGraphNNDescentNeighbor() = default;
            GeoGraphNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer)
            {
            }
            inline bool operator<(const GeoGraphNNDescentNeighbor &other) const
            {
                // return geo_distance_ < other.geo_distance_;
                return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
                // 较小的 geo_distance_ 值会被排序到较前的位置
            }
        };

        // struct GeoGraphNNDescentNeighbor_SkylineFurther
        // {
        //     unsigned id_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     bool flag;
        //     int layer_;
        //     GeoGraphNNDescentNeighbor() = default;
        //     GeoGraphNNDescentNeighbor(unsigned id, float emb_distance, float geo_distance, bool f, int layer) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(f), layer_(layer)
        //     {
        //     }
        //     inline bool operator<(const GeoGraphNNDescentNeighbor &other) const
        //     {
        //         // return geo_distance_ < other.geo_distance_;
        //         return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
        //         // 较小的 geo_distance_ 值会被排序到较前的位置
        //     }
        // };

        class GeoGraphNode
        {
        public:
            explicit GeoGraphNode(int id, int level, int max_m)
                : id_(id), level_(level), max_m_(max_m) //, friends_at_layer_(level + 1)
            {
                friends.reserve(max_m_ + 1);
                // for (int i = 0; i <= level; ++i)
                // friends_at_layer_[i].reserve(max_m_ + 1);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) { id_ = id; }
            inline int GetMaxM() const { return max_m_; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) { level_ = level; }
            inline void SetMaxM(int max_m) { max_m_ = max_m; }
            // inline GeoGraphNode *GetNode() const { return node_; }
            // inline std::vector<GeoGraphNeighbor> &GetFriends() { return friends_at_layer_[0]; }
            // inline std::vector<GeoGraphNeighbor> &GetFriends(int level) { return friends_at_layer_[level]; }
            inline std::vector<GeoGraphNeighbor> &GetFriends() { return friends; }

            // inline void SetFriends(int level, std::vector<GeoGraphNeighbor> &new_friends)
            // {
            //     if (level >= friends_at_layer_.size())
            //         friends_at_layer_.resize(level + 1);
            //     friends_at_layer_[level].swap(new_friends);
            // }

            inline void SetFriends(std::vector<GeoGraphNeighbor> &new_friends)
            {
                friends.swap(new_friends);
            }

            inline std::vector<GeoGraphSimpleNeighbor> &GetSearchFriends() { return friends_for_search; }

            // inline void SetFriends(int level, std::vector<GeoGraphNeighbor> &new_friends)
            // {
            //     if (level >= friends_at_layer_.size())
            //         friends_at_layer_.resize(level + 1);
            //     friends_at_layer_[level].swap(new_friends);
            // }

            inline void SetSearchFriends(std::vector<GeoGraphSimpleNeighbor> &new_friends)
            {
                friends_for_search.swap(new_friends);
            }

            inline std::mutex &GetAccessGuard() { return access_guard_; }

            // inline void AddVisitedNode(unsigned id) { Visited_Set.insert(id); }

            // inline bool NotVisited(unsigned id) {return Visited_Set.find(id) != Visited_Set.end();}

            // inline void AddFriends(GeoGraphEdge element, bool bCheckForDup)
            // {
            //     std::unique_lock<std::mutex> lock(access_guard_);
            //     if (bCheckForDup)
            //     {
            //         auto it = std::lower_bound(friends.begin(), friends.end(), element);
            //         // 使用std::lower_bound在friends_at_layer_[0]中搜索element的正确插入位置
            //         // 这个搜索过程假定friends_at_layer_[0]是预先排序的，以便能够有效地使用二分查找
            //         if (it == friends.end() || (it) != element)
            //         {
            //             friends.insert(it, element);
            //         }
            //     }
            //     else
            //     {
            //         friends.push_back(element);
            //     }
            // }
        private:
            int id_;
            int level_;
            size_t max_m_;
            std::vector<GeoGraphNeighbor> friends;
            std::vector<GeoGraphSimpleNeighbor> friends_for_search;
            // std::vector<std::vector<GeoGraphNeighbor>> friends_at_layer_;
            // std::unordered_set<unsigned> Visited_Set;
            std::mutex access_guard_;
        };

        // class GeoGraphEdge
        // {
        // public:
        //     GeoGraphEdge() = default;
        //     GeoGraphEdge(GeoGraphNode *node, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range) : node_{node}, emb_distance_(emb_distance), geo_distance_(geo_distance), available_range(range) {}
        //     // 移动构造函数
        //     GeoGraphEdge(GeoGraphEdge &&other) noexcept
        //         : node_{other.node_}, emb_distance_{other.emb_distance_}, geo_distance_{other.geo_distance_}, available_range(std::move(other.available_range))
        //     {
        //         other.node_ = nullptr;
        //     }
        //     // 移动赋值运算符
        //     GeoGraphEdge &operator=(GeoGraphEdge &&other) noexcept
        //     {
        //         if (this != &other)
        //         {
        //             node_ = other.node_;
        //             emb_distance_ = other.emb_distance_;
        //             geo_distance_ = other.geo_distance_;
        //             available_range = std::move(other.available_range);
        //             other.node_ = nullptr;
        //         }
        //         return *this;
        //     }
        //     // GeoGraphEdge(unsigned node_id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range):
        //     //     node_id_{node_id}, emb_distance_(emb_distance), geo_distance_(geo_distance), available_range(range) {}
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     // inline unsigned GetNodeID() const { return node_id_; }
        //     inline const std::vector<std::pair<float, float>> &GetRange() const { return available_range; }
        //     inline float GetEmbDistance() const { return emb_distance_; }
        //     inline float GetLocDistance() const { return geo_distance_; }

        // private:
        //     GeoGraphNode *node_;
        //     // unsigned node_id_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     std::vector<std::pair<float, float>> available_range;
        // };

        // struct skyline_naive
        // {
        //     std::mutex lock;                          // 互斥锁 用于并发访问控制
        //     std::vector<GeoGraphSimpleNeighbor> pool; // 存储邻居节点的优先队列 最小堆
        //     unsigned M;                               // 记录pool的最大大小, 也就是candidate边数

        //     std::vector<unsigned> nn_old;
        //     std::vector<unsigned> nn_new;
        //     std::vector<unsigned> rnn_old;
        //     std::vector<unsigned> rnn_new;

        //     skyline_naive() {}

        //     skyline_naive(unsigned l, unsigned s)
        //     {
        //         M = l;
        //         nn_new.resize(s * 2);
        //         nn_new.reserve(s * 2);
        //         // resize用于定义向量的实际大小（元素数量） 而reserve用于优化内存分配以应对向量大小的潜在增长
        //         pool.reserve(l);
        //     }

        //     skyline_naive(unsigned l, unsigned s, std::mt19937 &rng, unsigned N)
        //     {
        //         M = s;
        //         nn_new.resize(s * 2);
        //         GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N); // 还用随机数填充 nn_new
        //         nn_new.reserve(s * 2);
        //         pool.reserve(l);
        //     }

        //     skyline_naive(const skyline_naive &other)
        //     {
        //         M = other.M;
        //         std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
        //         nn_new.reserve(other.nn_new.capacity());
        //         pool.reserve(other.pool.capacity());
        //     }

        //     void findSkyline(std::vector<GeoGraphSimpleNeighbor> &points, std::vector<GeoGraphSimpleNeighbor> &skyline)
        //     {
        //         // Sort points by x-coordinate
        //         // Sweep to find skyline
        //         float max_emb_dis = std::numeric_limits<float>::max();
        //         for (const auto &point : points)
        //         {
        //             if (point.emb_distance_ < max_emb_dis)
        //             {
        //                 skyline.push_back(point);
        //                 max_emb_dis = point.emb_distance_;
        //             }
        //         }
        //         // O(n)
        //     }

        //     // 比较函数，用于构建最小堆
        //     // bool edist_priority(const GeoGraphSimpleNeighbor& a, const GeoGraphSimpleNeighbor& b) {
        //     //     return a.emb_distance_ > b.emb_distance_;  // 注意: '>' 使得堆成为最小堆
        //     // }

        //     void insert(unsigned id, float e_dist, float s_dist)
        //     {
        //         // 方法用于向 pool 中插入新的邻居节点，如果新节点的距离小于 pool 中最远节点的距离，则替换之
        //         //  pool 最大堆
        //         LockGuard guard(lock);

        //         for (unsigned i = 0; i < pool.size(); i++)
        //         {
        //             if (id == pool[i].id_)
        //                 return;
        //         }

        //         if (pool.size() < M)
        //         {
        //             pool.push_back(GeoGraphSimpleNeighbor(id, e_dist, s_dist, true));
        //             std::push_heap(pool.begin(), pool.end());
        //         }
        //         else
        //         {
        //             pool.push_back(GeoGraphSimpleNeighbor(id, e_dist, s_dist, true));
        //             std::push_heap(pool.begin(), pool.end());
        //             std::vector<GeoGraphSimpleNeighbor> tmp_pool;
        //             while (tmp_pool.size() < M && pool.size())
        //             {
        //                 std::vector<GeoGraphSimpleNeighbor> candidate;
        //                 findSkyline(pool, candidate);
        //                 auto iterCandidate = candidate.begin();
        //                 for (int i = 0; i < pool.size() && iterCandidate != candidate.end();)
        //                 {
        //                     if (pool[i].id_ == (*iterCandidate).id_)
        //                     {
        //                         pool.erase(pool.begin() + i);
        //                         tmp_pool.push_back((*iterCandidate));
        //                         ++iterCandidate;
        //                     }
        //                     else
        //                     {
        //                         i++;
        //                     }
        //                 }
        //             }
        //             tmp_pool.swap(pool);
        //         }
        //     }

        //     template <typename C>
        //     void join(C callback) const
        //     {
        //         for (unsigned const i : nn_new)
        //         {
        //             // 遍历 nn_new（新邻居）集合中的每个元素 i
        //             for (unsigned const j : nn_new)
        //             {
        //                 // 再次遍历 nn_new 集合中的每个元素 j
        //                 if (i < j)
        //                 {
        //                     callback(i, j);
        //                     // 如果 i 小于 j（避免重复处理和自连接），调用 callback 函数处理这对邻居节点
        //                 }
        //             }
        //             for (unsigned j : nn_old)
        //             {
        //                 callback(i, j);
        //                 // 接着遍历 nn_old（旧邻居）集合中的每个元素 j，对每个新旧邻居对调用 callback 函数
        //             }
        //         }
        //     }
        //     // 似乎是NN Descent
        // };

        // struct skyline_set
        // {
        //     std::mutex lock;
        //     std::vector<GeoGraphSimpleNeighbor> skyline_point;
        //     // 假设是按照skyline_的地理信息排序的
        //     skyline_set() {}
        //     skyline_set(std::vector<GeoGraphSimpleNeighbor> &skyline)
        //     {
        //         skyline_point.swap(skyline);
        //     }

        //     int insert(std::vector<GeoGraphSimpleNeighbor> &skyline)
        //     {
        //         skyline_point.swap(skyline);
        //         return 0;
        //     }

        //     int insert(unsigned id, float e_dist, float s_dist)
        //     {
        //         // three situation
        //         // insert into this skyline
        //         // dominate some points in the skyline
        //         // be dominated by some points in the skyline
        //         if (s_dist > skyline_point.back().geo_distance_ && e_dist > skyline_point.front().emb_distance_)
        //         {
        //             return -1;
        //             // the insert point is dominated by all the point in the existing skyline
        //             // not inserted
        //         }
        //         else if (s_dist < skyline_point.front().geo_distance_ && e_dist < skyline_point.back().emb_distance_)
        //         {
        //             return -2;
        //             // the insert point dominates all the points in the existing skyline
        //             // this can forms a better skyline
        //             // inserted into the pool
        //         }
        //         else
        //         {
        //             unsigned left = 0, right = skyline_point.size() - 1;
        //             unsigned K = right;
        //             if (skyline_point[left].geo_distance_ > s_dist)
        //             {
        //                 if (skyline_point[left].emb_distance_ >= e_dist)
        //                 {
        //                     return -2;
        //                     // the insert point dominates the first point
        //                     // it will be inserted
        //                 }
        //                 else if (skyline_point[left].emb_distance_ < e_dist)
        //                 {
        //                     // the insert point will be inserted into the left position
        //                     memmove((char *)&skyline_point[left + 1], &skyline_point[left], (K - left) * sizeof(GeoGraphSimpleNeighbor));
        //                     skyline_point[left] = GeoGraphSimpleNeighbor(id, e_dist, s_dist, true);
        //                     return left;
        //                 }
        //             }
        //             if (skyline_point[right].geo_distance_ < s_dist)
        //             {
        //                 if (skyline_point[right].emb_distance_ <= e_dist)
        //                 {
        //                     return -1;
        //                     // the insert point dominates the last point
        //                     // it will be inserted
        //                 }
        //                 else if (skyline_point[right].emb_distance_ > e_dist)
        //                 {
        //                     // the insert point will be inserted into the right position
        //                     skyline_point.emplace_back(GeoGraphSimpleNeighbor(id, e_dist, s_dist, true));
        //                     return right;
        //                 }
        //             }
        //             while (left < right - 1)
        //             {
        //                 int mid = (left + right) / 2;
        //                 if (skyline_point[mid].geo_distance_ > s_dist)
        //                     right = mid;
        //                 else
        //                     left = mid;
        //             }
        //             if (skyline_point[left].geo_distance_ < s_dist && skyline_point[left].emb_distance_ < e_dist)
        //                 return -1;
        //             if (s_dist < skyline_point[right].geo_distance_ && e_dist < skyline_point[right].emb_distance_)
        //                 return -2;
        //             memmove((char *)&skyline_point[right + 1], &skyline_point[right], (K - right) * sizeof(GeoGraphSimpleNeighbor));
        //             skyline_point[right] = GeoGraphSimpleNeighbor(id, e_dist, s_dist, true);
        //             return right;
        //         }
        //     }
        // };

        struct skyline_descent
        {
            std::mutex lock; // 互斥锁 用于并发访问控制
            // std::vector<std::vector<GeoGraphSimpleNeighbor>> pool;
            std::vector<GeoGraphNNDescentNeighbor> pool;
            std::vector<GeoGraphNNDescentNeighbor> outlier; // 存储邻居节点的优先队列 最小堆
            unsigned M;                                     // 记录pool的最大大小, 也就是candidate边数
            unsigned Q;                                     // 记录pool在update的过程中candidate set需要考虑的layer 也即quality result
            unsigned num_layer;
            unsigned use_range;
            // unsigned *pstart;
            std::unordered_set<unsigned> Visited_Set;
            // unsigned pool_size;
            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;
            skyline_descent() {}

            skyline_descent(unsigned l, unsigned s, unsigned q)
            {
                M = l;
                Q = q;
                // nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                // resize用于定义向量的实际大小（元素数量） 而reserve用于优化内存分配以应对向量大小的潜在增长
                pool.reserve(4 * l * l);
                // pool.reserve(4 * l * l);
                // pool_size = 0;
                // pstart = new unsigned[M](); // record the start node of each layer
            }

            skyline_descent(const skyline_descent &other)
            {
                M = other.M;
                Q = other.Q;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
                // nn_new.reserve(other.nn_new.capacity());
                // pool.reserve(other.pool.capacity());
            }
            // void findSkyline(std::vector<GeoGraphSimpleNeighbor> &points, std::vector<GeoGraphSimpleNeighbor> &skyline, std::vector<GeoGraphSimpleNeighbor> &remain_points)
            // {
            //     // Sort points by x-coordinate
            //     // Sweep to find skyline
            //     float max_emb_dis = std::numeric_limits<float>::max();
            //     for (const auto &point : points)
            //     {
            //         if (point.emb_distance_ < max_emb_dis)
            //         {
            //             skyline.push_back(point);
            //             max_emb_dis = point.emb_distance_;
            //         }
            //         else
            //         {
            //             remain_points.emplace_back(point);
            //         }
            //     }
            //     // O(n)
            // }

            // void init_neighor(std::vector<GeoGraphSimpleNeighbor> &insert_points)
            // {
            //     LockGuard guard(lock);
            //     std::vector<GeoGraphSimpleNeighbor> skyline_result;
            //     std::vector<GeoGraphSimpleNeighbor> remain_points;
            //     int l = 0;
            //     while (insert_points.size())
            //     {
            //         findSkyline(insert_points, skyline_result, remain_points);
            //         insert_points.swap(remain_points);
            //         pool.push_back(skyline_result);
            //         std::vector<GeoGraphSimpleNeighbor>().swap(skyline_result);
            //         std::vector<GeoGraphSimpleNeighbor>().swap(remain_points);
            //         l++;
            //     }
            //     for (int i = 0; i < l; i++)
            //     {
            //         for (int j = 0; j < pool[i].size(); j++)
            //         {
            //             pool_id_map[pool[i][j].id_] = true;
            //         }
            //     }
            //     std::vector<unsigned>().swap(nn_new);
            //     std::vector<unsigned>().swap(nn_old);
            //     std::vector<unsigned>().swap(rnn_new);
            //     std::vector<unsigned>().swap(rnn_old);
            // }

            void findSkyline(std::vector<GeoGraphNNDescentNeighbor> &points, std::vector<GeoGraphNNDescentNeighbor> &skyline, std::vector<GeoGraphNNDescentNeighbor> &remain_points)
            {
                // Sort points by x-coordinate
                // Sweep to find skyline
                float max_emb_dis = std::numeric_limits<float>::max();
                for (const auto &point : points)
                {
                    if (point.emb_distance_ < max_emb_dis)
                    {
                        skyline.push_back(point);
                        max_emb_dis = point.emb_distance_;
                    }
                    else
                    {
                        remain_points.emplace_back(point);
                    }
                }
                // O(n)
            }

            void init_neighor(std::vector<GeoGraphNNDescentNeighbor> &insert_points)
            {
                LockGuard guard(lock);
                std::vector<GeoGraphNNDescentNeighbor> skyline_result;
                std::vector<GeoGraphNNDescentNeighbor> remain_points;
                int l = 0;
                while (!insert_points.empty())
                {
                    findSkyline(insert_points, skyline_result, remain_points);
                    insert_points.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, true, l);
                    }
                    outlier.swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(remain_points);
                    l++;
                    // if (l >= Q)
                    // {
                    //     break;
                    // }
                }
                num_layer = l;
                sort(pool.begin(), pool.end());
                std::vector<unsigned>().swap(nn_new);
                std::vector<unsigned>().swap(nn_old);
                std::vector<unsigned>().swap(rnn_new);
                std::vector<unsigned>().swap(rnn_old);
            }

            void updateNeighbor()
            {
                LockGuard guard(lock);
                std::vector<GeoGraphNNDescentNeighbor> skyline_result;
                std::vector<GeoGraphNNDescentNeighbor> remain_points;
                std::vector<GeoGraphNNDescentNeighbor> candidate;
                candidate.clear();
                candidate.swap(pool);
                int l = 0;
                sort(candidate.begin(), candidate.end());
                while (pool.size() < M && candidate.size() > 0)
                {
                    findSkyline(candidate, skyline_result, remain_points);
                    candidate.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.flag, l);
                    }
                    outlier.swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(remain_points);
                    l++;
                    // if (l >= Q)
                    // {
                    //     break;
                    // }
                }
                num_layer = l;
            }

            // void insert(std::vector<GeoGraphSimpleNeighbor> &skyline_vector, std::vector<GeoGraphSimpleNeighbor> &been_dominated_objects, bool &inserted, unsigned id, float e_dist, float s_dist)
            // {
            //     // this insert try to insert a node and replicate find skyline results
            //     for (int i = 0; i < skyline_vector.size(); i++)
            //     {
            //         if (skyline_vector[i].geo_distance_ < s_dist)
            //         {
            //             if (skyline_vector[i].emb_distance_ <= e_dist)
            //             {
            //                 return;
            //             }
            //         }
            //         else
            //         {
            //             // skyline_vector[i].geo_distance_ >= s_dist
            //             if (skyline_vector[i].emb_distance_ <= e_dist)
            //             {
            //                 // then just inserted it
            //                 skyline_vector.insert(skyline_vector.begin() + i, GeoGraphSimpleNeighbor(id, e_dist, s_dist, true));
            //                 inserted = true;
            //                 return;
            //             }
            //             else
            //             {
            //                 inserted = true;
            //                 for (size_t j = i; j < skyline_vector.size(); ++j)
            //                 {
            //                     if (skyline_vector[j].emb_distance_ > e_dist)
            //                     {
            //                         been_dominated_objects.push_back(skyline_vector[j]);
            //                     }
            //                     else
            //                     {
            //                         break; // 不符合条件，终止收集
            //                     }
            //                 }
            //                 skyline_vector.erase(skyline_vector.begin() + i, skyline_vector.begin() + i + been_dominated_objects.size());
            //                 skyline_vector.insert(skyline_vector.begin() + i, GeoGraphSimpleNeighbor(id, e_dist, s_dist, true));
            //                 return;
            //             }
            //         }
            //     }
            // }

            // void insert(std::vector<GeoGraphSimpleNeighbor> &skyline_vector, std::vector<GeoGraphSimpleNeighbor> &inserted_objects)
            // {
            //     // this insert try to insert a node and replicate find skyline results
            //     std::vector<GeoGraphSimpleNeighbor> been_dominated_objects;
            //     float pesudo_s_dist = inserted_objects.front().geo_distance_;
            //     float pesudo_e_dist = inserted_objects.back().emb_distance_;
            //     for (int i = 0; i < skyline_vector.size(); i++)
            //     {
            //         if (skyline_vector[i].geo_distance_ < pesudo_s_dist)
            //         {
            //             if (skyline_vector[i].emb_distance_ <= pesudo_e_dist)
            //             {
            //                 std::cout << "error for inserting vector of objects in line 1026" << std::endl;
            //                 exit(1);
            //             }
            //             else
            //             {
            //                 continue;
            //             }
            //         }
            //         else
            //         {
            //             // skyline_vector[i].geo_distance_ >= s_dist
            //             if (skyline_vector[i].emb_distance_ <= pesudo_e_dist)
            //             {
            //                 // then just inserted it
            //                 skyline_vector.insert(skyline_vector.begin() + i, inserted_objects.begin(), inserted_objects.end());
            //                 break;
            //             }
            //             else
            //             {
            //                 int begin = i;
            //                 // Collect dominated objects
            //                 while (i < skyline_vector.size() && skyline_vector[i].emb_distance_ > pesudo_e_dist)
            //                 {
            //                     been_dominated_objects.push_back(skyline_vector[i]);
            //                     ++i;
            //                 }
            //                 // Remove the dominated objects and insert new ones
            //                 skyline_vector.erase(skyline_vector.begin() + begin, skyline_vector.begin() + i);
            //                 skyline_vector.insert(skyline_vector.begin() + begin, inserted_objects.begin(), inserted_objects.end());
            //                 break;
            //             }
            //             break;
            //         }
            //     }
            //     inserted_objects.swap(been_dominated_objects);
            //     return;
            // }
            // void insert(unsigned id, float e_dist, float s_dist)
            // {
            //     this is very slow
            //     // 方法用于向 pool 中插入新的邻居节点，如果新节点的距离小于 pool 中最远节点的距离，则替换之
            //     //  pool 最大堆
            //     LockGuard guard(lock);
            //     // for (unsigned i = 0; i < pool.size(); i++)
            //     // {
            //     //     if (id == pool[i].id_)
            //     //         return;
            //     // }
            //     if (pool_id_map[id])
            //     {
            //         return;
            //     }
            //     bool inserted = false;
            //     std::vector<GeoGraphSimpleNeighbor> been_dominated_objects;
            //     int total_size = 0;
            //     for (int i = 0; i < pool.size(); i++)
            //     {
            //         if (inserted == true)
            //         {
            //             if (been_dominated_objects.size() == 0)
            //             {
            //                 break;
            //             }
            //             else
            //             {
            //                 insert(pool[i], been_dominated_objects);
            //             }
            //         }
            //         else
            //         {
            //             insert(pool[i], been_dominated_objects, inserted, id, e_dist, s_dist);
            //         }
            //         total_size = total_size + pool[i].size();
            //         if (total_size >= M)
            //         {
            //             pool.resize(i + 1);
            //         }
            //     }

            //     if (been_dominated_objects.size() != 0 && inserted && total_size < M)
            //     {
            //         pool.push_back(been_dominated_objects);
            //     }
            // }

            // void insert(unsigned id, float e_dist, float s_dist)
            // {
            //     LockGuard guard(lock);
            //     int insert_index = -1;
            //     if (pool.size() >= M && num_layer > 1)
            //     {
            //         std::vector<int> deleteIndices;
            //         for (int i = pool.size() - 1; i >= 0; i--)
            //         {
            //             if (pool[i].layer_ >= num_layer - 1)
            //             {
            //                 pool.erase(pool.begin() + i);
            //             }
            //         }
            //         num_layer--;
            //     }
            //     int not_been_dominated_layer = num_layer, been_dominated_layer = -1;
            //     GeoGraphNNDescentNeighbor inserted_neighbor = GeoGraphNNDescentNeighbor(id, e_dist, s_dist, true, -1);
            //     std::vector<std::vector<int>> been_dominated_nodes(num_layer);

            //     for (int i = 0; i < pool.size(); i++)
            //     {
            //         if (pool[i].id_ == id)
            //         {
            //             return; // 如果找到重复的ID，直接返回
            //         }

            //         if (pool[i].geo_distance_ < s_dist)
            //         {
            //             if (pool[i].emb_distance_ <= e_dist)
            //             {
            //                 been_dominated_layer = std::max(been_dominated_layer, pool[i].layer_);
            //             }
            //             if (been_dominated_layer == num_layer - 1)
            //             {
            //                 return;
            //             }
            //         }
            //         else
            //         {
            //             not_been_dominated_layer = been_dominated_layer + 1;
            //             if (insert_index == -1)
            //             {
            //                 insert_index = i; // 记录插入位置
            //             }
            //             if (pool[i].layer_ >= not_been_dominated_layer && pool[i].emb_distance_ >= e_dist) // here is the error 3
            //             {
            //                 been_dominated_nodes[pool[i].layer_].emplace_back(i);
            //             }
            //         }
            //     }

            //     not_been_dominated_layer = been_dominated_layer + 1;
            //     if (insert_index == -1)
            //     {
            //         insert_index = pool.size();
            //     }

            //     inserted_neighbor.layer_ = not_been_dominated_layer;
            //     float prune_s_dist = s_dist;
            //     float prune_e_dist = e_dist;
            //     for (int i = not_been_dominated_layer; i < been_dominated_nodes.size(); i++)
            //     {
            //         bool replaced = false;
            //         int start_move_id = -1;
            //         int end_move_id = -1;
            //         if (been_dominated_nodes[i].size() == 0)
            //             break;
            //         else
            //         {
            //             for (int j = 0; j < been_dominated_nodes[i].size(); j++)
            //             {
            //                 if (pool[been_dominated_nodes[i][j]].geo_distance_ < prune_s_dist)
            //                 {
            //                     continue;
            //                 }
            //                 else if (pool[been_dominated_nodes[i][j]].emb_distance_ < prune_e_dist)
            //                 {
            //                     break;
            //                 }
            //                 else
            //                 {
            //                     if (start_move_id == -1)
            //                     {
            //                         start_move_id = been_dominated_nodes[i][j];
            //                     }
            //                     end_move_id = been_dominated_nodes[i][j];
            //                     pool[end_move_id].layer_++;
            //                     replaced = true;
            //                     if (pool[end_move_id].layer_ == num_layer)
            //                     {
            //                         num_layer++;
            //                     }
            //                 }
            //             }
            //         }
            //         if (!replaced)
            //         {
            //             break;
            //         }
            //         prune_s_dist = pool[start_move_id].geo_distance_;
            //         prune_e_dist = pool[end_move_id].emb_distance_;
            //     }
            //     pool.insert(pool.begin() + insert_index, inserted_neighbor);
            //     return;
            // }

            // void insert(unsigned id, float e_dist, float s_dist)
            // {
            //     // this tries to maintain multiple layer skyline results
            //     LockGuard guard(lock);
            //     // the fasest so far
            //     if (Visited_Set.find(id) != Visited_Set.end())
            //     {
            //         return;
            //     }
            //     Visited_Set.insert(id);

            //     int insert_index = -1;
            //     if (pool.size() >= M && num_layer > 1)
            //     {
            //         for (int i = pool.size() - 1; i >= 0; i--)
            //         {
            //             if (pool[i].layer_ >= num_layer - 1)
            //             {
            //                 pool.erase(pool.begin() + i);
            //             }
            //         }
            //         num_layer--;
            //     }

            //     int not_been_dominated_layer = num_layer, been_dominated_layer = -1;

            //     GeoGraphNNDescentNeighbor inserted_neighbor = GeoGraphNNDescentNeighbor(id, e_dist, s_dist, true, -1);
            //     std::vector<std::vector<int>> been_dominated_nodes(num_layer);

            //     int left = 0;
            //     int right = pool.size() - 1;
            //     if (pool[left].geo_distance_ > s_dist)
            //     {
            //         // insert position
            //         left = -1;
            //         right = 0;
            //     }

            //     if (pool[right].geo_distance_ < s_dist)
            //     {
            //         // insert position
            //         left = right;
            //         right++;
            //     }

            //     while (left < right - 1)
            //     {
            //         int mid = (left + right) / 2;
            //         if (pool[mid].geo_distance_ < s_dist)
            //             left = mid;
            //         else
            //             right = mid;
            //     }

            //     while (left >= 0)
            //     {
            //         if (pool[left].layer_ > been_dominated_layer)
            //         {
            //             if (pool[left].emb_distance_ <= e_dist)
            //             {
            //                 been_dominated_layer = std::max(been_dominated_layer, pool[left].layer_);
            //             }
            //             if (been_dominated_layer == num_layer - 1)
            //                 return;
            //         }
            //         left--;
            //     }
            //     not_been_dominated_layer = been_dominated_layer + 1;
            //     insert_index = right;
            //     while (right < pool.size())
            //     {
            //         if (pool[right].layer_ >= not_been_dominated_layer && pool[right].emb_distance_ >= e_dist)
            //         {
            //             been_dominated_nodes[pool[right].layer_].emplace_back(right);
            //         }
            //         right++;
            //     }
            //     inserted_neighbor.layer_ = not_been_dominated_layer;
            //     float prune_s_dist = s_dist;
            //     float prune_e_dist = e_dist;
            //     for (int i = not_been_dominated_layer; i < been_dominated_nodes.size(); i++)
            //     {
            //         bool replaced = false;
            //         int start_move_id = -1;
            //         int end_move_id = -1;
            //         if (been_dominated_nodes[i].size() == 0)
            //             break;
            //         else
            //         {
            //             for (int j = 0; j < been_dominated_nodes[i].size(); j++)
            //             {
            //                 if (pool[been_dominated_nodes[i][j]].geo_distance_ < prune_s_dist)
            //                 {
            //                     continue;
            //                 }
            //                 else if (pool[been_dominated_nodes[i][j]].emb_distance_ < prune_e_dist)
            //                 {
            //                     break;
            //                 }
            //                 else
            //                 {
            //                     if (start_move_id == -1)
            //                     {
            //                         start_move_id = been_dominated_nodes[i][j];
            //                     }
            //                     end_move_id = been_dominated_nodes[i][j];
            //                     pool[end_move_id].layer_++;
            //                     replaced = true;
            //                     if (pool[end_move_id].layer_ == num_layer)
            //                     {
            //                         num_layer++;
            //                     }
            //                     else if (pool[end_move_id].layer_ > num_layer)
            //                     {
            //                         std::cout << "line 1430 error" << std::endl;
            //                         exit(1);
            //                     }
            //                 }
            //             }
            //         }
            //         if (!replaced)
            //         {
            //             break;
            //         }
            //         prune_s_dist = pool[start_move_id].geo_distance_;
            //         prune_e_dist = pool[end_move_id].emb_distance_;
            //     }
            //     pool.insert(pool.begin() + insert_index, inserted_neighbor);
            //     return;
            // }

            void insert(unsigned id, float e_dist, float s_dist)
            {
                LockGuard guard(lock);
                // the simplest not very slow
                if (Visited_Set.find(id) != Visited_Set.end())
                {
                    return;
                }
                Visited_Set.insert(id);
                for (int i = 0; i < outlier.size(); i++)
                {
                    if (outlier[i].emb_distance_ <= e_dist && outlier[i].geo_distance_ <= s_dist)
                    {
                        return;
                    }
                    else if (outlier[i].geo_distance_ > s_dist)
                    {
                        break;
                    }
                }
                pool.emplace_back(id, e_dist, s_dist, true, -1);
                return;
            }

            template <typename C>
            void join(C callback) const
            {
                for (unsigned const i : nn_new)
                {
                    // 遍历 nn_new（新邻居）集合中的每个元素 i
                    for (unsigned const j : nn_new)
                    {
                        // 再次遍历 nn_new 集合中的每个元素 j
                        if (i < j)
                        {
                            callback(i, j);
                            // 如果 i 小于 j（避免重复处理和自连接），调用 callback 函数处理这对邻居节点
                        }
                    }
                    for (unsigned j : nn_old)
                    {
                        // callback(i, j, insert_delete_time, search_time, compute_time_complexity, mtx);
                        // 接着遍历 nn_old（旧邻居）集合中的每个元素 j，对每个新旧邻居对调用 callback 函数
                        callback(i, j);
                    }
                }
            }
            // 似乎是NN Descent
        };

        // class GeoGraphNNDescentNeighborListNode
        // {
        // public:
        //     int id_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     int layer_;
        //     bool flag;
        //     std::shared_ptr<GeoGraphNNDescentNeighbor> next_in_next_List;
        //     std::shared_ptr<GeoGraphNNDescentNeighborListNode> prev;
        //     std::shared_ptr<GeoGraphNNDescentNeighborListNode> next;
        //     GeoGraphNNDescentNeighborListNode() = default;
        //     GeoGraphNNDescentNeighborListNode(unsigned id, float emb_distance, float geo_distance, bool flag_, int l) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), flag(flag_), layer_(l)
        //     {
        //         next_in_next_List = NULL;
        //         prev = NULL;
        //         next = NULL;
        //     }
        // };

        struct skyline_queue
        {
            std::vector<GeoGraphNNDescentNeighbor> pool;
            unsigned M; // 记录pool的最大大小, 也就是candidate边数
            unsigned num_layer;
            skyline_queue() {}

            skyline_queue(unsigned l)
            {
                M = l;
                pool.reserve(M);
            }

            skyline_queue(const skyline_queue &other)
            {
                M = other.M;
                pool.reserve(other.pool.capacity());
            }

            float cross(const GeoGraphNNDescentNeighbor &O, const GeoGraphNNDescentNeighbor &A, const GeoGraphNNDescentNeighbor &B)
            {
                return (A.geo_distance_ - O.geo_distance_) * (B.emb_distance_ - O.emb_distance_) - (A.emb_distance_ - O.emb_distance_) * (B.geo_distance_ - O.geo_distance_);
            }

            void findConvexHull(std::vector<GeoGraphNNDescentNeighbor> &points, std::vector<GeoGraphNNDescentNeighbor> &convex_hull, std::vector<GeoGraphNNDescentNeighbor> &remain_points)
            {
                // Build the lower hull
                for (const auto &point : points)
                {
                    while (convex_hull.size() >= 2 && cross(convex_hull[convex_hull.size() - 2], convex_hull.back(), point) <= 0)
                    {
                        remain_points.push_back(convex_hull.back());
                        convex_hull.pop_back();
                    }
                    convex_hull.push_back(point);
                }
                // Remove the last point of the upper hull because it's the same as the first point of the lower hull
            }

            void init_queue(std::vector<GeoGraphNNDescentNeighbor> &insert_points)
            {
                std::vector<GeoGraphNNDescentNeighbor> skyline_result;
                std::vector<GeoGraphNNDescentNeighbor> remain_points;
                int l = 0;
                while (!insert_points.empty())
                {
                    findSkyline(insert_points, skyline_result, remain_points);
                    // findConvexHull(insert_points, skyline_result, remain_points);
                    insert_points.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, true, l);
                    }
                    std::vector<GeoGraphNNDescentNeighbor>().swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(remain_points);
                    l++;
                }
                num_layer = l;
            }

            void findSkyline(std::vector<GeoGraphNNDescentNeighbor> &points, std::vector<GeoGraphNNDescentNeighbor> &skyline, std::vector<GeoGraphNNDescentNeighbor> &remain_points)
            {
                // Sort points by x-coordinate
                // Sweep to find skyline
                float min_emb_dis = std::numeric_limits<float>::max();
                for (const auto &point : points)
                {
                    if (point.emb_distance_ < min_emb_dis)
                    {
                        skyline.push_back(point);
                        min_emb_dis = point.emb_distance_;
                    }
                    else
                    {
                        remain_points.emplace_back(point);
                    }
                }
                // O(n)
            }

            void updateNeighbor(int &nk)
            {
                std::vector<GeoGraphNNDescentNeighbor> skyline_result;
                std::vector<GeoGraphNNDescentNeighbor> remain_points;
                std::vector<GeoGraphNNDescentNeighbor> candidate;
                candidate.swap(pool);
                int l = 0;
                int k = 0;
                sort(candidate.begin(), candidate.end());
                bool updated = true;
                while (pool.size() < M && candidate.size() > 0)
                {
                    findSkyline(candidate, skyline_result, remain_points);
                    // findConvexHull(candidate, skyline_result, remain_points); // too slow
                    candidate.swap(remain_points);
                    for (auto &point : skyline_result)
                    {
                        pool.emplace_back(point.id_, point.emb_distance_, point.geo_distance_, point.flag, l);
                        if (updated)
                        {
                            if (point.flag == true)
                            {
                                nk = k;
                                updated = false;
                            }
                            else
                            {
                                nk++;
                            }
                        }
                        k++;
                    }
                    std::vector<GeoGraphNNDescentNeighbor>().swap(skyline_result);
                    std::vector<GeoGraphNNDescentNeighbor>().swap(remain_points);
                    l++;
                }
                num_layer = l;
            }
        };
        // struct skyline_queue
        // {
        //     std::vector<std::shared_ptr<GeoGraphNNDescentNeighborListNode>> pool;
        //     std::vector<std::shared_ptr<GeoGraphNNDescentNeighborListNode>> tail;
        //     unsigned M; // 记录pool的最大大小, 也就是candidate边数
        //     unsigned num_layer;
        //     skyline_queue() {}

        //     skyline_queue(unsigned l)
        //     {
        //         M = l;
        //         pool.reserve(M);
        //         tail.reserve(M);
        //     }

        //     skyline_queue(const skyline_queue &other)
        //     {
        //         M = other.M;
        //         pool.reserve(other.pool.capacity());
        //     }

        //     void init_neighor(std::vector<GeoGraphNNDescentNeighbor> &points)
        //     {
        //         std::vector<GeoGraphNNDescentNeighbor> skyline_result;
        //         std::vector<GeoGraphNNDescentNeighbor> remain_points;
        //         int l = 0;
        //         while (!points.empty())
        //         {
        //             findSkyline(points, skyline_result, remain_points);
        //             points.swap(remain_points);
        //             std::shared_ptr<GeoGraphNNDescentNeighborListNode> iter = pool[l];
        //             for (auto &point : skyline_result)
        //             {
        //                 auto node = std::make_shared<GeoGraphNNDescentNeighborListNode>(point.id_, point.emb_distance_, point.geo_distance_, true, l);
        //                 if (pool[l] == nullptr)
        //                 {
        //                     pool[l] = node; // 如果当前层级的第一个节点为空，将第一个节点指针指向当前节点
        //                 }
        //                 else
        //                 {
        //                     tail[l]->next = node;
        //                     node->prev = tail[l];
        //                 }
        //                 tail[l] = node;
        //             }
        //             std::vector<GeoGraphNNDescentNeighbor>().swap(skyline_result);
        //             std::vector<GeoGraphNNDescentNeighbor>().swap(remain_points);
        //             l++;
        //         }
        //         num_layer = l;
        //         for (int i = 0; i < num_layer - 1; i++)
        //         {
        //             auto it_i = pool[i];
        //             auto it_next = pool[i + 1];
        //             while (it_i != NULL && it_next != NULL)
        //             {
        //                 if (it_next->geo_distance_ > it_i->geo_distance_)
        //                 {
        //                     it_i->next_in_next_List = it_next;
        //                     it_i = it_i->next;
        //                 }
        //                 else
        //                 {
        //                     it_next = it_next->next;
        //                 }
        //             }
        //         }
        //     }

        //     void findSkyline(std::vector<GeoGraphNNDescentNeighbor> &points, std::vector<GeoGraphNNDescentNeighbor> &skyline, std::vector<GeoGraphNNDescentNeighbor> &remain_points)
        //     {
        //         // Sort points by x-coordinate
        //         // Sweep to find skyline
        //         float max_emb_dis = std::numeric_limits<float>::max();
        //         for (const auto &point : points)
        //         {
        //             if (point.emb_distance_ < max_emb_dis)
        //             {
        //                 skyline.push_back(point);
        //                 max_emb_dis = point.emb_distance_;
        //             }
        //             else
        //             {
        //                 remain_points.emplace_back(point);
        //             }
        //         }
        //         // O(n)
        //     }

        //     int insert(unsigned id, float e_dist, float s_dist)
        //     {
        //         int l = 0;
        //         std::shared_ptr<GeoGraphNNDescentNeighborListNode> it = pool[0];
        //         std::shared_ptr<GeoGraphNNDescentNeighborListNode> prev_it = nullptr;
        //         while (l < num_layer)
        //         {
        //             // Find the correct insertion point
        //             while (it != nullptr && it->geo_distance_ < s_dist)
        //             {
        //                 prev_it = it;
        //                 it = it->next;
        //             }

        //             // Check if the current node can dominate the new node
        //             if (prev_it != nullptr && prev_it->geo_distance_ < s_dist && prev_it->emb_distance_ < e_dist)
        //             {
        //                 // If we can't insert here, move to the next layer if possible
        //                 l++;
        //                 if (l >= num_layer)
        //                 {
        //                     return false; // Exit if no more layers are available
        //                 }
        //                 it = (prev_it->next_in_next_List) ? prev_it->next_in_next_List : tail[l];
        //                 prev_it = (it) ? it->prev : tail[l]; // Set prev_it to it->prev if it is not nullptr, otherwise to tail[l]
        //                 continue;
        //             }

        //             // Create and insert the new node
        //             auto new_node = std::make_shared<GeoGraphNNDescentNeighborListNode>(id, e_dist, s_dist, true, l);
        //             if (prev_it == nullptr)
        //             {
        //                 // Insert at the head
        //                 pool[l] = new_node;
        //                 new_node->next = it;
        //                 if (it != nullptr)
        //                 {
        //                     it->prev = new_node;
        //                 }
        //             }
        //             else
        //             { // Insert in the middle or end
        //                 new_node->next = prev_it->next;
        //                 new_node->prev = prev_it;
        //                 prev_it->next = new_node;
        //                 if (new_node->next != nullptr)
        //                 {
        //                     new_node->next->prev = new_node;
        //                 }
        //             }

        //             std::shared_ptr<GeoGraphNNDescentNeighborListNode> current = new_node->next;
        //             while (current != nullptr && new_node->geo_distance_ <= current->geo_distance_ && new_node->emb_distance_ <= current->emb_distance_)
        //             {
        //                 // This node is dominated, move it to the appropriate position in the next layer
        //                 auto dominated_node_begin = current;
        //                 current = current->next; // Move forward in the list

        //                 // Remove dominated node from the current layer
        //                 if (dominated_node->prev)
        //                 {
        //                     dominated_node->prev->next = dominated_node->next;
        //                 }
        //                 if (dominated_node->next)
        //                 {
        //                     dominated_node->next->prev = dominated_node->prev;
        //                 }

        //                 // Re-insert dominated node in the next layer if possible
        //                 if (l + 1 < num_layer)
        //                 {
        //                     insertDominatedNode(l + 1, dominated_node);
        //                 }
        //             }
        //             if (it == nullptr && new_node->next == nullptr)
        //             { // Update tail if necessary
        //                 tail[l] = new_node;
        //             }

        //             return l; // Insertion successful
        //         }

        //         return -1; // Insertion failed in all layers
        //     }
        // };
        typedef std::vector<skyline_descent> SkylineGeoGraph;
        SkylineGeoGraph skylinegeograph_;

        // typedef std::vector<std::vector<GeoGraphSimpleNeighbor>> InitGeoGraphGraph;
        // InitGeoGraphGraph initgeograph;

        // InitGeoGraphGraph &GetinitGeoGraph()
        // {
        //     return initgeograph;
        // }

        class GeoGraph_Convex
        {
        public:
            GeoGraph_Convex(GeoGraphNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline GeoGraphNode *GetNode() const { return node_; }

        private:
            GeoGraphNode *node_;
            float emb_distance_;
            float geo_distance_;
        };

        template <typename KeyType, typename DataType>
        class MaxHeap
        {
        public:
            class Item
            {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType &key) : key(key) {}
                Item(const KeyType &key, const DataType &data) : key(key), data(data) {}
                bool operator<(const Item &i2) const
                {
                    return key < i2.key; // 修改这里以适应最大堆，父节点的键应大于等于子节点的键
                }
            };

            MaxHeap()
            {
                std::make_heap(v_.begin(), v_.end()); // 确保v_初始化为一个堆
            }

            const KeyType top_key()
            {
                if (v_.empty())                                                                     // 优化：使用empty()检查是否为空
                    throw std::runtime_error("[Error] Called top_key() operation with empty heap"); // 无元素时抛出异常
                return v_[0].key;
            }

            Item top()
            {
                if (v_.empty())                                                                 // 使用empty()检查是否为空
                    throw std::runtime_error("[Error] Called top() operation with empty heap"); // 无元素时抛出异常
                return v_[0];
            }
            // 返回堆顶元素 最大元素

            void pop()
            {
                if (v_.empty())                                                                 // 添加检查避免在空堆上操作
                    throw std::runtime_error("[Error] Called pop() operation with empty heap"); // 无元素时抛出异常
                std::pop_heap(v_.begin(), v_.end());                                            // 将最大元素移到末尾
                v_.pop_back();                                                                  // 移除末尾元素（原最大元素）
            }
            // 移除堆顶元素

            void push(const KeyType &key, const DataType &data)
            {
                v_.emplace_back(Item(key, data));     // 添加新元素
                std::push_heap(v_.begin(), v_.end()); // 重新调整堆
            }
            // 向堆中添加新元素

            size_t size()
            {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class GeoGraph_FurtherFirst
        {
        public:
            GeoGraph_FurtherFirst(GeoGraphNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetDistance() const { return dist_; }
            inline GeoGraphNode *GetNode() const { return node_; }
            bool operator<(const GeoGraph_FurtherFirst &n) const
            {
                return (dist_ < n.GetDistance());
                // 距离较小的节点会被视为“优先级较小” 因此FurtherFirst用于构建最大堆
            }

        private:
            GeoGraphNode *node_;
            float emb_distance_;
            float geo_distance_;
            float dist_;
        };

        class GeoGraph_CloserFirst
        {
        public:
            GeoGraph_CloserFirst(GeoGraphNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance), dist_(dist) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline float GetDistance() const { return dist_; }
            inline GeoGraphNode *GetNode() const { return node_; }
            bool operator<(const GeoGraph_CloserFirst &n) const
            {
                return (dist_ > n.GetDistance());
                // 距离较小的节点会被视为“优先级较高” 因此CloserFirst用于构建最小堆
            }

        private:
            GeoGraphNode *node_;
            float emb_distance_;
            float geo_distance_;
            float dist_;
        };

        class GeoGraph_GeoFurtherFirst
        {
        public:
            GeoGraph_GeoFurtherFirst(GeoGraphNode *node, float emb_distance, float geo_distance, float dist) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline GeoGraphNode *GetNode() const { return node_; }
            bool operator<(const GeoGraph_GeoFurtherFirst &n) const
            {
                return (geo_distance_ < n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ < n.GetEmbDistance()));
                // 距离较小的节点会被视为“优先级较小” 因此 FurtherFirst 用于构建最大堆
            }

        private:
            GeoGraphNode *node_;
            float emb_distance_;
            float geo_distance_;
        };

        class GeoGraph_GeoCloserFirst
        {
        public:
            GeoGraph_GeoCloserFirst(GeoGraphNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
            inline float GetEmbDistance() const { return emb_distance_; }
            inline float GetLocDistance() const { return geo_distance_; }
            inline GeoGraphNode *GetNode() const { return node_; }
            bool operator<(const GeoGraph_GeoCloserFirst &n) const
            {
                return (geo_distance_ > n.GetLocDistance() || (geo_distance_ == n.GetLocDistance() && emb_distance_ > n.GetEmbDistance()));
                // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
            }

        private:
            GeoGraphNode *node_;
            float emb_distance_;
            float geo_distance_;
        };

        GeoGraphNode *geograph_enterpoint_ = nullptr;
        std::vector<GeoGraphNode *> geograph_nodes_;
        std::vector<GeoGraphNode *> geograph_enterpoints;
        std::vector<GeoGraphNNDescentNeighbor> geograph_enterpoints_skyeline;

        std::mutex enterpoint_mutex;
        std::vector<unsigned> enterpoint_set;
        unsigned rnn_size;
        float *emb_center, *loc_center;
        // class Emb_FurtherFirst
        // {
        // public:
        //     Emb_FurtherFirst(GeoGraphNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
        //     inline float GetEmbDistance() const { return emb_distance_; }
        //     inline float GetLocDistance() const { return geo_distance_; }
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     bool operator<(const Emb_FurtherFirst &n) const
        //     {
        //         return (emb_distance_ < n.GetEmbDistance());
        //     }
        // private:
        //     GeoGraphNode *node_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        // };

        // class Geo_CloserFirst
        // {
        // public:
        //     Geo_CloserFirst(GeoGraphNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
        //     inline float GetEmbDistance() const { return emb_distance_; }
        //     inline float GetLocDistance() const { return geo_distance_; }
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     bool operator<(const Geo_CloserFirst &n) const
        //     {
        //         return (geo_distance_ > n.GetLocDistance());
        //     }
        // private:
        //     GeoGraphNode *node_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        // };

        // class Emb_CloserFirst
        // {
        // public:
        //     Emb_CloserFirst(GeoGraphNode *node, float emb_distance, float geo_distance) : node_(node), emb_distance_(emb_distance), geo_distance_(geo_distance) {}
        //     inline float GetEmbDistance() const { return emb_distance_; }
        //     inline float GetLocDistance() const { return geo_distance_; }
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     bool operator<(const Emb_CloserFirst &n) const
        //     {
        //         return (emb_distance_ > n.GetEmbDistance());
        //     }
        // private:
        //     GeoGraphNode *node_;
        //     float emb_distance_;
        //     float geo_distance_;
        //     // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        // };

        // class GeoGraphFurtherFirst
        // {
        // public:
        //     GeoGraphFurtherFirst(GeoGraphNode *node, float distance) : node_(node), distance_(distance) {}
        //     inline float GetDistance() const { return distance_; }
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     bool operator<(const GeoGraphFurtherFirst &n) const
        //     {
        //         return (distance_ < n.GetDistance());
        //     }

        // private:
        //     GeoGraphNode *node_;
        //     float distance_;
        //     // 距离较小的节点会被视为“优先级较低” 因此 FurtherFirst 实际上是用于构建最大堆的
        // };

        // class GeoGraphCloserFirst
        // {
        // public:
        //     GeoGraphCloserFirst(GeoGraphNode *node, float distance) : node_(node), distance_(distance) {}
        //     inline float GetDistance() const { return distance_; }
        //     inline GeoGraphNode *GetNode() const { return node_; }
        //     bool operator<(const GeoGraphCloserFirst &n) const
        //     {
        //         return (distance_ > n.GetDistance());
        //     }

        // private:
        //     GeoGraphNode *node_;
        //     float distance_;
        //     // 距离较小的节点会被视为“优先级较高” 因此 CloserFirst 用于构建最小堆
        // };

        // struct GeoGraphSimpleNeighbor
        // {
        //     unsigned id_;
        //     float distance_;
        //     // float geo_distance;

        //     GeoGraphSimpleNeighbor() = default;
        //     GeoGraphSimpleNeighbor(unsigned id, float distance) : id_{id}, distance_{distance} {}

        //     inline bool operator<(const GeoGraphSimpleNeighbor &other) const
        //     {
        //         return distance_ > other.distance_;
        //     }
        // };

        // typedef typename std::pair<GeoGraphNode *, float> Geo_IdDistancePair;
        // struct GeoIdDistancePairMinHeapComparer
        // {
        //     bool operator()(const Geo_IdDistancePair &p1, const Geo_IdDistancePair &p2) const
        //     {
        //         return p1.second > p2.second;
        //     }
        // };
        // typedef typename boost::heap::d_ary_heap<Geo_IdDistancePair, boost::heap::arity<4>, boost::heap::compare<GeoIdDistancePairMinHeapComparer>> GeoIdDistancePairMinHeap;

        // class GeoSubArea{
        //     // Mbr mbr;
        //     public:
        //         std::vector<unsigned> objects_;
        //         unsigned original_size = 0;
        //         float* center_coord = new float[2]();
        //         GeoGraphNode *GeoSubGraph_enterpoint_ = nullptr;
        //         std::vector<GeoGraphNode *> GeoSubGraph_nodes_;

        //         GeoSubArea(std::vector<unsigned> &objects, float *baseLocData)
        //         {
        //             center_coord[0] = 0;
        //             center_coord[1] = 0;
        //             for (int i = 0; i < objects.size(); i++)
        //             {
        //                 objects_.emplace_back(objects[i]);
        //                 center_coord[0] = center_coord[0] + baseLocData[objects[i] * 2];
        //                 center_coord[1] = center_coord[1] + baseLocData[objects[i] * 2 + 1];
        //             }
        //             original_size = objects_.size();
        //             center_coord[0] = center_coord[0] / objects.size();
        //             center_coord[1] = center_coord[1] / objects.size();
        //         }

        //         inline std::vector<unsigned> &GetObjects() { return objects_; }
        //         inline void add_object(unsigned object_id){ objects_.push_back(object_id);}
        // };
    };

    class Index : public NNDescent, public NSW, public HNSW, public SSG, public NSG, public GEOGRAPH
    {
    public:
        explicit Index(float max_emb_dist, float max_spatial_dist)
        {
            e_dist_ = new E_Distance(max_emb_dist);
            s_dist_ = new E_Distance(max_spatial_dist);
            // max_emb_dist_ = max_emb_dist;
            // max_spatial_dist_ = max_spatial_dist;
        }

        ~Index()
        {
            delete e_dist_;
            delete s_dist_;
        }

        struct SimpleNeighbor
        {
            unsigned id;
            float distance;

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

            inline bool operator<(const SimpleNeighbor &other) const
            {
                return distance < other.distance;
                // 该优先队列实际上表现为一个最大堆
            }
        };

        // sorted
        typedef std::vector<std::vector<SimpleNeighbor>> FinalGraph;
        typedef std::vector<std::vector<unsigned>> LoadGraph;

        class VisitedList
        {
        public:
            VisitedList(unsigned size) : size_(size), mark_(1)
            {
                visited_ = new unsigned int[size_];
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }

            ~VisitedList() { delete[] visited_; }

            inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }

            inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }

            inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }

            inline void Reset()
            {
                if (++mark_ == 0)
                {
                    mark_ = 1;
                    memset(visited_, 0, sizeof(unsigned int) * size_);
                }
            }

            inline unsigned int *GetVisited() { return visited_; }

            inline unsigned int GetVisitMark() { return mark_; }

        private:
            unsigned int *visited_;
            unsigned int size_;
            unsigned int mark_;
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn)
        {
            // find the location to insert
            // 数组 addr 的大小由参数 K 指定
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance)
            {
                memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance)
            {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1)
            {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)
                    right = mid;
                else
                    left = mid;
            }
            // check equal ID

            while (left > 0)
            {
                if (addr[left].distance < nn.distance)
                    break;
                if (addr[left].id == nn.id)
                    return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)
                return K + 1;
            memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        float *getBaseEmbData() const
        {
            return base_emb_data_;
        }

        void setBaseEmbData(float *baseEmbData)
        {
            base_emb_data_ = baseEmbData;
        }

        float *getBaseLocData() const
        {
            return base_loc_data_;
        }

        void setBaseLocData(float *baseLocData)
        {
            base_loc_data_ = baseLocData;
        }

        float *getQueryEmbData() const
        {
            return query_emb_data_;
        }

        void setQueryEmbData(float *queryEmbData)
        {
            query_emb_data_ = queryEmbData;
        }

        float *getQueryLocData() const
        {
            return query_loc_data_;
        }

        void setQueryLocData(float *queryLocData)
        {
            query_loc_data_ = queryLocData;
        }

        unsigned int *getGroundData() const
        {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData)
        {
            ground_data_ = groundData;
        }

        unsigned int getBaseLen() const
        {
            return base_len_;
        }

        void setBaseLen(unsigned int baseLen)
        {
            base_len_ = baseLen;
        }

        unsigned int getQueryLen() const
        {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen)
        {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const
        {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen)
        {
            ground_len_ = groundLen;
        }

        unsigned int getBaseEmbDim() const
        {
            return base_emb_dim_;
        }

        unsigned int getBaseLocDim() const
        {
            return base_loc_dim_;
        }

        void setBaseEmbDim(unsigned int baseEmbDim)
        {
            base_emb_dim_ = baseEmbDim;
        }

        void setBaseLocDim(unsigned int baseLocDim)
        {
            base_loc_dim_ = baseLocDim;
        }

        unsigned int getQueryEmbDim() const
        {
            return query_emb_dim_;
        }

        unsigned int getQueryLocDim() const
        {
            return query_loc_dim_;
        }

        void setQueryEmbDim(unsigned int queryEmbDim)
        {
            query_emb_dim_ = queryEmbDim;
        }

        void setQueryLocDim(unsigned int queryLocDim)
        {
            query_loc_dim_ = queryLocDim;
        }

        unsigned int getGroundDim() const
        {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim)
        {
            ground_dim_ = groundDim;
        }

        Parameters &getParam()
        {
            return param_;
        }

        void setParam(const Parameters &param)
        {
            param_ = param;
        }

        unsigned int getInitEdgesNum() const
        {
            return init_edges_num;
        }

        void setInitEdgesNum(unsigned int initEdgesNum)
        {
            init_edges_num = initEdgesNum;
        }

        unsigned int getCandidatesEdgesNum() const
        {
            return candidates_edges_num;
        }

        unsigned int getUpdateLayerNum() const
        {
            return update_layer_num;
        }

        void setCandidatesEdgesNum(unsigned int candidatesEdgesNum)
        {
            candidates_edges_num = candidatesEdgesNum;
        }

        void setUpdateLayerNum(unsigned int updatelayernum)
        {
            update_layer_num = updatelayernum;
        }

        unsigned int getResultEdgesNum() const
        {
            return result_edges_num;
        }

        void setResultEdgesNum(unsigned int resultEdgesNum)
        {
            result_edges_num = resultEdgesNum;
        }

        void set_alpha(float alpha)
        {
            // alpha_ = alpha;
            param_.set<unsigned>("alpha", alpha);
        }

        float get_alpha() const
        {
            return param_.get<float>("alpha");
        }
        E_Distance *get_E_Dist() const
        {
            return e_dist_;
        }

        // void Init_R_Tree(float max_s_dist)
        // {
        //     rtree_index = new RTreeIndex(max_s_dist);
        // }

        RTreeIndex &get_R_Tree()
        {
            return rtree_index;
        }

        E_Distance *get_S_Dist() const
        {
            return s_dist_;
        }

        FinalGraph &getFinalGraph()
        {
            return final_graph_;
        }

        SkylineGeoGraph &getSkylineGraph()
        {
            return skylinegeograph_;
        }

        LoadGraph &getLoadGraph()
        {
            return load_graph_;
        }

        LoadGraph &getExactGraph()
        {
            return exact_graph_;
        }

        TYPE getCandidateType() const
        {
            return candidate_type;
        }

        void setCandidateType(TYPE candidateType)
        {
            candidate_type = candidateType;
        }

        TYPE getPruneType() const
        {
            return prune_type;
        }

        void setPruneType(TYPE pruneType)
        {
            prune_type = pruneType;
        }

        TYPE getEntryType() const
        {
            return entry_type;
        }

        void setEntryType(TYPE entryType)
        {
            entry_type = entryType;
        }

        void setConnType(TYPE connType)
        {
            conn_type = connType;
        }

        TYPE getConnType() const
        {
            return conn_type;
        }

        unsigned int getDistCount() const
        {
            return dist_count;
        }

        void resetDistCount()
        {
            dist_count = 0;
        }

        void addDistCount()
        {
            dist_count += 1;
        }

        unsigned int getHopCount() const
        {
            return hop_count;
        }

        void resetHopCount()
        {
            hop_count = 0;
        }

        void addHopCount()
        {
            hop_count += 1;
        }

        void setNumThreads(const unsigned numthreads)
        {
            omp_set_num_threads(numthreads);
        }

        int i = 0;
        bool debug = false;

    private:
        float *base_emb_data_, *base_loc_data_, *query_emb_data_, *query_loc_data_;
        unsigned *ground_data_;

        unsigned base_len_, query_len_, ground_len_;
        unsigned base_emb_dim_, base_loc_dim_, query_emb_dim_, query_loc_dim_, ground_dim_;

        Parameters param_;
        unsigned init_edges_num;       // S
        unsigned candidates_edges_num; // L
        unsigned result_edges_num;     // K
        unsigned update_layer_num;

        E_Distance *e_dist_;
        E_Distance *s_dist_;

        FinalGraph final_graph_;
        LoadGraph load_graph_;
        LoadGraph exact_graph_;

        TYPE entry_type;
        TYPE candidate_type;
        TYPE prune_type;
        TYPE conn_type;

        RTreeIndex rtree_index;

        unsigned dist_count = 0;
        unsigned hop_count = 0;

        float alpha_;
        float max_emb_dist_, max_spatial_dist_;
    };
}

#endif

// if (pool.size() >= M)
// {
//     for (auto it = pool.begin(); it != pool.end();)
//     {
//         if (it->layer_ >= num_layer - 1)
//         {
//             it = pool.erase(it);
//         }
//         else
//         {
//             ++it;
//         }
//     }
//     num_layer--;
// }

// std::vector<std::vector<std::list<GeoGraphNNDescentNeighbor>::iterator>> been_dominated_nodes(num_layer);

// for (auto it = pool.begin(); it != pool.end(); it++)
// {
//     if (it->id_ == id)
//     {
//         return; // 如果找到重复的ID，直接返回
//     }

//     if (it->geo_distance_ < s_dist)
//     {
//         if (it->emb_distance_ <= e_dist)
//         {
//             been_dominated_layer = std::max(been_dominated_layer, it->layer_);
//         }
//         if (been_dominated_layer == num_layer - 1)
//         {
//             return;
//         }
//     }
//     else
//     {
//         not_been_dominated_layer = been_dominated_layer + 1;
//         if (insert_idx == pool.end())
//         {
//             insert_idx = it; // 记录插入位置
//         }
//         if (it->layer_ >= not_been_dominated_layer && it->emb_distance_ >= e_dist)
//         {
//             been_dominated_nodes[it->layer_].emplace_back(it);
//         }
//     }
// }
// not_been_dominated_layer = been_dominated_layer + 1;
// inserted_neighbor.layer_ = not_been_dominated_layer;
// float prune_s_dist = s_dist;
// float prune_e_dist = e_dist;
// for (int i = not_been_dominated_layer; i < been_dominated_nodes.size(); i++)
// {
//     bool replaced = false;
//     auto start_move_id = pool.end();
//     auto end_move_id = pool.end();
//     if (been_dominated_nodes[i].size() == 0)
//         break;
//     else
//     {
//         for (int j = 0; j < been_dominated_nodes[i].size(); j++)
//         {
//             if (been_dominated_nodes[i][j]->geo_distance_ < prune_s_dist)
//             {
//                 continue;
//             }
//             else if (been_dominated_nodes[i][j]->emb_distance_ < prune_e_dist)
//             {
//                 break;
//             }
//             else
//             {
//                 if (start_move_id == pool.end())
//                 {
//                     start_move_id = been_dominated_nodes[i][j];
//                 }
//                 end_move_id = been_dominated_nodes[i][j];
//                 end_move_id->layer_++;
//                 replaced = true;
//                 if (end_move_id->layer_ == num_layer)
//                 {
//                     num_layer++;
//                 }
//             }
//         }
//     }
//     if (!replaced)
//     {
//         break;
//     }
//     prune_s_dist = start_move_id->geo_distance_;
//     prune_e_dist = end_move_id->emb_distance_;
// }
// pool.insert(insert_idx, inserted_neighbor);
// return;