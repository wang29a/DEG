#include "component.h"

namespace stkq
{

    /**
     * NN-Descent Refine
     */
    void ComponentRefineNNDescent::RefineInner()
    {

        // L ITER S R
        SetConfigs();

        init();

        NNDescent();

        // graph_ -> final_graph
#pragma omp parallel for
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            std::vector<Index::SimpleNeighbor> tmp;
            tmp.reserve(index->getCandidatesEdgesNum());

            std::sort(index->graph_[i].pool.begin(), index->graph_[i].pool.end());

            for (auto &j : index->graph_[i].pool)
            {
                tmp.push_back(Index::SimpleNeighbor(j.id, j.distance));
            }

            index->getFinalGraph()[i].swap(tmp);

            std::vector<Index::Neighbor>().swap(index->graph_[i].pool);
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
        }

        std::vector<Index::nhood>().swap(index->graph_);

        unsigned range = index->getResultEdgesNum();

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * range];

        // PRUNE
        std::cout << "__PRUNE : NAIVE__" << std::endl;
        auto *b = new ComponentPruneNaive(index);

#ifdef PARALLEL
#pragma omp parallel
#endif
        {
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
#ifdef PARALLEL
#pragma omp for schedule(dynamic, 100)
#endif
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                b->PruneInner(n, range, flags, index->getFinalGraph()[n], cut_graph_);
            }
        }

        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            Index::SimpleNeighbor *src_pool = cut_graph_ + n * range;
            int len = 0;
            for (unsigned i = 0; i < range; i++)
            {
                if (src_pool[i].distance == -1)
                    break;
                len++;
                index->getFinalGraph()[n][i] = src_pool[i];
            }
            index->getFinalGraph()[n].resize(len);
        }

        delete[] cut_graph_;
    }

    void ComponentRefineNNDescent::SetConfigs()
    {
        index->setCandidatesEdgesNum(index->getParam().get<unsigned>("L")); // 50
        // 此参数通常设置 NN-Descent 算法中要考虑的候选边的数量
        // 它确定在细化过程中从中选择最近邻居的池的大小
        index->setResultEdgesNum(index->getParam().get<unsigned>("K")); // 20
        // 该参数设置结果边的数量
        // 它本质上定义了最终图中邻域的大小，确定细化过程后每个点应有多少个最近邻
        index->R = index->getParam().get<unsigned>("R"); // 100
        // 此参数可能会影响 NN-Descent 算法中考虑的随机邻居的数量
        // 它可以通过引入随机性来使邻居选择过程多样化
        index->ITER = index->getParam().get<unsigned>("ITER"); // 6
        // 该参数设置 NN-Descent 算法的迭代次数。它确定算法将迭代数据以细化邻域图的次数
    }

    void ComponentRefineNNDescent::init()
    {
        index->graph_.reserve(index->getBaseLen());

        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            index->graph_.emplace_back(Index::nhood(index->getCandidatesEdgesNum(), index->getInitEdgesNum()));
            // nhood(unsigned l, unsigned s)
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            for (unsigned j = 0; j < index->getFinalGraph()[i].size(); j++)
            {
                Index::SimpleNeighbor node = index->getFinalGraph()[i][j];
                index->graph_[i].pool.emplace_back(Index::Neighbor(node.id, node.distance, true));
                // true代表着这个neighbor是新加入的neighbor
            }
            std::make_heap(index->graph_[i].pool.begin(), index->graph_[i].pool.end());
            // 在C++中，std::make_heap 是一个标准库函数，它用于将一个序列 通常是一个容器，如 vector 转换成一个堆 heap
            index->graph_[i].pool.reserve(index->getCandidatesEdgesNum());
        }
    }

    void ComponentRefineNNDescent::NNDescent()
    {
        for (unsigned it = 0; it < index->ITER; it++)
        {
            auto s = std::chrono::high_resolution_clock::now();
            join();
            auto e = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            std::cout << "Initialization time for join one iter: " << duration << " milliseconds" << std::endl;
            // join是从neighbor的neighbor来更新pool
            update();
        }
    }

    void ComponentRefineNNDescent::join()
    {
// default(shared)：这将并行区域中变量的默认数据共享属性设置为共享，这意味着所有线程都可以访问相同的数据
// schedule(dynamic, 100)：这指定循环迭代如何在线程之间划分。dynamic意味着每个线程请求一个新的工作100是每个块的大小。通过dynamic调度，工作负载可以得到平衡，因为提前完成的线程可以请求新工作
#ifdef PARALLEL
#pragma omp parallel for default(shared) schedule(dynamic, 100)
#endif
        for (unsigned n = 0; n < index->getBaseLen(); n++)
        {
            // 遍历所有节点
            //  对每个节点的邻居执行 join 操作
            //  [&] 是一个 lambda 表达式的捕获子句 用于指定 lambda 表达式如何捕获外部作用域中的变量
            //  *按引用捕获**：这意味着 lambda 表达式内部可以访问并修改外部作用域中的变量。它不会创建这些变量的副本，而是直接引用外部作用域中的实际变量
            index->graph_[n].join([&](unsigned i, unsigned j)
                                  {
                //这里使用了lambda表达式定义了 join 操作
                if (i != j) {
                    float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + i * index->getBaseEmbDim(),
                                                            index->getBaseEmbData() + j * index->getBaseEmbDim(),
                                                            index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + i * index->getBaseLocDim(),
                                                            index->getBaseLocData() + j * index->getBaseLocDim(),
                                                            index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    index->graph_[i].insert(j, dist);
                    index->graph_[j].insert(i, dist);
                } });
        }
    }

    void ComponentRefineNNDescent::update()
    {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); i++)
        {
            std::vector<unsigned>().swap(index->graph_[i].nn_new);
            std::vector<unsigned>().swap(index->graph_[i].nn_old);
            // 清空每个节点的新邻居和旧邻居集合
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            auto &nn = index->graph_[n];
            std::sort(nn.pool.begin(), nn.pool.end());
            // 对每个节点的邻居池进行排序
            // 调整大小并保留所需的数量
            if (nn.pool.size() > index->getCandidatesEdgesNum())
            {
                nn.pool.resize(index->getCandidatesEdgesNum());
            }
            // 遍历每个节点，根据当前的邻居池更新新旧邻居集合
            // 从 nn.pool 的尾部删除多余的元素
            nn.pool.reserve(index->getCandidatesEdgesNum());
            unsigned maxl = std::min(nn.M + index->getInitEdgesNum(), (unsigned)nn.pool.size());
            // unsigned maxl = nn.pool.size();
            // nn.M = S
            // 计算最大邻居数量
            // 计算对每个节点来说，理想的邻居数量（maxl）。这是节点当前邻居数量 (nn.M) 加上初始边数 (getInitEdgesNum()) 和实际邻居池大小的较小值
            unsigned c = 0;
            unsigned l = 0;
            while ((l < maxl) && (c < index->getInitEdgesNum())) // this is very critical for speeding up
            // while (l < maxl)
            {
                // 遍历每个节点的邻居池，以更新邻居信息
                if (nn.pool[l].flag)
                {
                    ++c; // 这里更新新加入的neighbor c
                }
                ++l;
            }
            nn.M = l; // 更新节点的邻居数量 M 为 l，这是考虑到了新旧邻居的综合数量
        }
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned n = 0; n < index->getBaseLen(); ++n)
        {
            auto &nnhd = index->graph_[n];
            auto &nn_new = nnhd.nn_new;
            auto &nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.M; ++l)
            {
                auto &nn = nnhd.pool[l];              // 获取邻居SimpleNeighbor
                auto &nhood_o = index->graph_[nn.id]; // 根据SimpleNeighbor的id得到对应的nhood
                if (nn.flag)
                {
                    nn_new.push_back(nn.id);
                    // if (nn.distance > nhood_o.pool.back().distance)
                    // {
                    // nn是n的邻居
                    // nhood_o是nn的邻居，也就是n的邻居的邻居
                    // 如果nn.distance比nhood_o.pool.back().distance, 也就是nhood_o的最大的邻居大, 那么就在nhood_o.rnn_new中加入n
                    LockGuard guard(nhood_o.lock);
                    if (nhood_o.rnn_new.size() < index->R)
                        nhood_o.rnn_new.push_back(n);
                    else
                    {
                        unsigned int pos = rand() % index->R;
                        nhood_o.rnn_new[pos] = n;
                    }
                    // }
                    nn.flag = false;
                }
                else
                {
                    nn_old.push_back(nn.id);
                    // if (nn.distance > nhood_o.pool.back().distance) // i think this is a error
                    // {
                    LockGuard guard(nhood_o.lock);
                    if (nhood_o.rnn_old.size() < index->R)
                        nhood_o.rnn_old.push_back(n);
                    else
                    {
                        unsigned int pos = rand() % index->R;
                        nhood_o.rnn_old[pos] = n;
                    }
                    // }
                }
            }
            std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
        }
        float nn_new_size = 0, nn_old_size = 0;
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (unsigned i = 0; i < index->getBaseLen(); ++i)
        {
            auto &nn_new = index->graph_[i].nn_new;
            auto &nn_old = index->graph_[i].nn_old;
            auto &rnn_new = index->graph_[i].rnn_new;
            auto &rnn_old = index->graph_[i].rnn_old;
            if (index->R && rnn_new.size() > index->R)
            {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(index->R);
            }
            // 如果反向新邻居列表（rnn_new）的大小超过了预设的限制（R），则将其随机打乱并裁剪到 R 的大小
            // 将处理过的反向新邻居列表添加到新邻居列表中
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            nn_new_size = nn_new_size + nn_new.size();
            if (index->R && rnn_old.size() > index->R)
            {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(index->R);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if (nn_old.size() > index->R * 2)
            {
                nn_old.resize(index->R * 2);
                nn_old.reserve(index->R * 2);
            }
            nn_old_size = nn_old_size + nn_old.size();
            std::vector<unsigned>().swap(index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(index->graph_[i].rnn_old);
        }
        std::cout << "nn_new_size: " << nn_new_size / index->getBaseLen() << "\t"
                  << "nn_old_size: " << nn_old_size / index->getBaseLen() << std::endl;
    }

    /**
     * NSG Refine :
     *  Entry     : Centroid
     *  CANDIDATE : GREEDY(NSG)
     *  PRUNE     : NSG
     *  CONN      : DFS
     */
    void ComponentRefineNSG::RefineInner()
    {
        SetConfigs();
        // ENTRY
        std::cout << "__ENTRY : Centroid__" << std::endl;
        auto *a = new ComponentRefineEntryCentroid(index);
        a->EntryInner();
        std::cout << "__ENTRY : FINISH" << std::endl;

        auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t)index->R_refine];
        Link(cut_graph_);
        index->getFinalGraph().resize(index->getBaseLen());

        for (size_t i = 0; i < index->getBaseLen(); i++)
        {
            Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t)index->R_refine;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < index->R_refine; j++)
            {
                if (pool[j].distance == -1)
                    break;
                pool_size = j;
            }
            pool_size++;
            index->getFinalGraph()[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++)
            {
                index->getFinalGraph()[i][j].id = pool[j].id;
                index->getFinalGraph()[i][j].distance = pool[j].distance;
            }
        }
        // CONN
        std::cout << "__CONN : DFS__" << std::endl;
        auto *c = new ComponentConnNSGDFS(index);
        c->ConnInner();
    }

    void ComponentRefineNSG::SetConfigs()
    {
        index->R_refine = index->getParam().get<unsigned>("R_refine");
        index->L_refine = index->getParam().get<unsigned>("L_refine");
        index->C_refine = index->getParam().get<unsigned>("C_refine");
        index->width = index->R_refine;
    }

    void ComponentRefineNSG::Link(Index::SimpleNeighbor *cut_graph_)
    {
        std::vector<std::mutex> locks(index->getBaseLen());

        // CANDIDATE
        std::cout << "__CANDIDATE : GREEDY(NSG)__" << std::endl;
        auto *a = new ComponentCandidateNSG(index);

        // PRUNE
        std::cout << "__PRUNE : NSG__" << std::endl;
        auto *b = new ComponentPruneNSG(index);

#pragma omp parallel
        {
            std::vector<Index::SimpleNeighbor> pool;
            boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                pool.clear();
                flags.reset();
                a->CandidateInner(n, index->ep_, flags, pool);
                b->PruneInner(n, index->R_refine, flags, pool, cut_graph_);
            }
            std::vector<Index::SimpleNeighbor>().swap(pool);

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < index->getBaseLen(); ++n)
            {
                InterInsert(n, index->R_refine, locks, cut_graph_);
            }
        }
    }

    void ComponentRefineNSG::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks,
                                         Index::SimpleNeighbor *cut_graph_)
    {
        Index::SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++)
        {
            if (src_pool[i].distance == -1)
                break;

            Index::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            Index::SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

            std::vector<Index::SimpleNeighbor> temp_pool;
            int dup = 0;
            {
                LockGuard guard(locks[des]);
                for (size_t j = 0; j < range; j++)
                {
                    if (des_pool[j].distance == -1)
                        break;
                    if (n == des_pool[j].id)
                    {
                        dup = 1;
                        break;
                    }
                    temp_pool.push_back(des_pool[j]);
                }
            }
            if (dup)
                continue;

            temp_pool.push_back(sn);
            if (temp_pool.size() > range)
            {
                std::vector<Index::SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size())
                {
                    auto &p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++)
                    {
                        if (p.id == result[t].id)
                        {
                            occlude = true;
                            break;
                        }

                        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)result[t].id * index->getBaseEmbDim(),
                                                                 index->getBaseEmbData() + (size_t)p.id * index->getBaseEmbDim(),
                                                                 index->getBaseEmbDim());

                        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)result[t].id * index->getBaseLocDim(),
                                                                 index->getBaseLocData() + (size_t)p.id * index->getBaseLocDim(),
                                                                 index->getBaseLocDim());

                        float djk = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                        if (djk < p.distance /* dik */)
                        {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude)
                        result.push_back(p);
                }
                {
                    LockGuard guard(locks[des]);
                    for (unsigned t = 0; t < result.size(); t++)
                    {
                        des_pool[t] = result[t];
                    }
                }
            }
            else
            {
                LockGuard guard(locks[des]);
                for (unsigned t = 0; t < range; t++)
                {
                    if (des_pool[t].distance == -1)
                    {
                        des_pool[t] = sn;
                        if (t + 1 < range)
                            des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    /**
     * NSSG Refine :
     *  Entry      : Centroid
     *  CANDIDATE  : PROPAGATION 2
     *  PRUNE      : NSSG
     *  CONN       : DFS_Expand
     */
    //     void ComponentRefineSSG::RefineInner() {
    //         SetConfigs();

    //         // ENTRY
    //         // auto *a = new ComponentRefineEntryCentroid(index);
    //         // a->EntryInner();

    //         auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->R_refine];
    //         Link(cut_graph_);

    //         index->getFinalGraph().resize(index->getBaseLen());

    //         for (size_t i = 0; i < index->getBaseLen(); i++) {
    //             Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->R_refine;
    //             unsigned pool_size = 0;
    //             for (unsigned j = 0; j < index->R_refine; j++) {
    //                 if (pool[j].distance == -1) break;
    //                 pool_size = j;
    //             }
    //             pool_size++;
    //             index->getFinalGraph()[i].resize(pool_size);
    //             for (unsigned j = 0; j < pool_size; j++) {
    //                 Index::SimpleNeighbor nn(pool[j].id, pool[j].distance);
    //                 index->getFinalGraph()[i][j] = nn;
    //             }
    //         }

    //         // CONN
    //         std::cout << "__CONN : DFS__" << std::endl;
    //         auto *c = new ComponentConnSSGDFS(index);

    //         c->ConnInner();
    //     }

    //     void ComponentRefineSSG::SetConfigs() {
    //         index->R_refine = index->getParam().get<unsigned>("R_refine");
    //         index->L_refine = index->getParam().get<unsigned>("L_refine");
    //         index->A = index->getParam().get<float>("A");
    //         index->n_try = index->getParam().get<unsigned>("n_try");

    //         index->width = index->R_refine;
    //     }

    //     void ComponentRefineSSG::Link(Index::SimpleNeighbor *cut_graph_) {
    //         /*
    //          std::cerr << "Graph Link" << std::endl;
    //          unsigned progress = 0;
    //          unsigned percent = 100;
    //          unsigned step_size = nd_ / percent;
    //          std::mutex progress_lock;
    //          */
    //         std::vector<std::mutex> locks(index->getBaseLen());

    //         // CANDIDATE
    //         std::cout << "__CANDIDATE : PROPAGATION 2__" << std::endl;
    //         ComponentCandidate *a = new ComponentCandidatePropagation2(index);

    //         // PRUNE
    //         std::cout << "__PRUNE : NSSG__" << std::endl;
    //         ComponentPrune *b = new ComponentPruneSSG(index);

    // #pragma omp parallel
    //         {
    //             // unsigned cnt = 0;
    //             std::vector<Index::SimpleNeighbor> pool;
    //             boost::dynamic_bitset<> flags{index->getBaseLen(), 0};

    // #pragma omp for schedule(dynamic, 100)
    //             for (unsigned n = 0; n < index->getBaseLen(); ++n) {
    //                 pool.clear();
    //                 flags.reset();

    //                 a->CandidateInner(n, n, flags, pool);
    //                 //std::cout << "candidate : " << pool.size() << std::endl;

    //                 b->PruneInner(n, index->R_refine, flags, pool, cut_graph_);
    //                 //std::cout << "prune : " << pool.size() << std::endl;

    //                 /*
    //                 cnt++;
    //                 if (cnt % step_size == 0) {
    //                   LockGuard g(progress_lock);
    //                   std::cout << progress++ << "/" << percent << " completed" << std::endl;
    //                 }
    //                 */
    //             }
    //         }

    //         double kPi = std::acos(-1);
    //         float threshold = std::cos(index->A / 180 * kPi);
    // #pragma omp parallel
    // #pragma omp for schedule(dynamic, 100)
    //         for (unsigned n = 0; n < index->getBaseLen(); ++n) {
    //             InterInsert(n, index->R_refine, threshold, locks, cut_graph_);
    //         }
    //     }

    //     void ComponentRefineSSG::InterInsert(unsigned n, unsigned range, float threshold, std::vector<std::mutex> &locks,
    //                                          Index::SimpleNeighbor *cut_graph_) {
    //         Index::SimpleNeighbor *src_pool = cut_graph_ + (size_t) n * (size_t) range;
    //         for (size_t i = 0; i < range; i++) {
    //             if (src_pool[i].distance == -1) break;

    //             Index::SimpleNeighbor sn(n, src_pool[i].distance);
    //             size_t des = src_pool[i].id;
    //             Index::SimpleNeighbor *des_pool = cut_graph_ + des * (size_t) range;

    //             std::vector<Index::SimpleNeighbor> temp_pool;
    //             int dup = 0;
    //             {
    //                 Index::LockGuard guard(locks[des]);
    //                 for (size_t j = 0; j < range; j++) {
    //                     if (des_pool[j].distance == -1) break;
    //                     if (n == des_pool[j].id) {
    //                         dup = 1;
    //                         break;
    //                     }
    //                     temp_pool.push_back(des_pool[j]);
    //                 }
    //             }
    //             if (dup) continue;

    //             temp_pool.push_back(sn);
    //             if (temp_pool.size() > range) {
    //                 std::vector<Index::SimpleNeighbor> result;
    //                 unsigned start = 0;
    //                 std::sort(temp_pool.begin(), temp_pool.end());
    //                 result.push_back(temp_pool[start]);
    //                 while (result.size() < range && (++start) < temp_pool.size()) {
    //                     auto &p = temp_pool[start];
    //                     bool occlude = false;
    //                     for (unsigned t = 0; t < result.size(); t++) {
    //                         if (p.id == result[t].id) {
    //                             occlude = true;
    //                             break;
    //                         }
    //                         float djk = index->getDist()->compare(
    //                                 index->getBaseData() + index->getBaseDim() * (size_t) result[t].id,
    //                                 index->getBaseData() + index->getBaseDim() * (size_t) p.id,
    //                                 (unsigned) index->getBaseDim());
    //                         float cos_ij = (p.distance + result[t].distance - djk) / 2 /
    //                                        sqrt(p.distance * result[t].distance);
    //                         if (cos_ij > threshold) {
    //                             occlude = true;
    //                             break;
    //                         }
    //                     }
    //                     if (!occlude) result.push_back(p);
    //                 }
    //                 {
    //                     Index::LockGuard guard(locks[des]);
    //                     for (unsigned t = 0; t < result.size(); t++) {
    //                         des_pool[t] = result[t];
    //                     }
    //                     if (result.size() < range) {
    //                         des_pool[result.size()].distance = -1;
    //                     }
    //                 }
    //             } else {
    //                 Index::LockGuard guard(locks[des]);
    //                 for (unsigned t = 0; t < range; t++) {
    //                     if (des_pool[t].distance == -1) {
    //                         des_pool[t] = sn;
    //                         if (t + 1 < range) des_pool[t + 1].distance = -1;
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    /**
     * DPG Refine :
     *  Entry     : Centroid
     *  CANDIDATE : PROPAGATION 2
     *  PRUNE     : NSSG
     *  CONN      : DFS_Expand
     */
    //     void ComponentRefineDPG::RefineInner() {
    //         SetConfigs();
    //         auto *cut_graph_ = new Index::SimpleNeighbor[index->getBaseLen() * (size_t) index->L_dpg];
    //         Link(cut_graph_);

    //         index->getFinalGraph().resize(index->getBaseLen());

    //         for (size_t i = 0; i < index->getBaseLen(); i++) {
    //             Index::SimpleNeighbor *pool = cut_graph_ + i * (size_t) index->L_dpg;
    //             unsigned pool_size = 0;
    //             for (unsigned j = 0; j < index->L_dpg; j++) {
    //                 if (pool[j].distance == -1) break;
    //                 pool_size = j;
    //             }
    //             pool_size++;
    //             index->getFinalGraph()[i].resize(pool_size);
    //             for (unsigned j = 0; j < pool_size; j++) {
    //                 Index::SimpleNeighbor nn(pool[j].id, pool[j].distance);
    //                 index->getFinalGraph()[i][j] = nn;
    //             }
    //         }

    //         // CONN
    //         // std::cout << "__CONN : DFS__" << std::endl;
    //         auto *c = new ComponentConnReverse(index);
    //         c->ConnInner();
    //     }

    //     void ComponentRefineDPG::SetConfigs() {
    //         index->K = index->getParam().get<unsigned>("K");
    //         index->L = index->getParam().get<unsigned>("L");
    //         index->S = index->getParam().get<unsigned>("S");
    //         index->R = index->getParam().get<unsigned>("R");
    //         index->ITER = index->getParam().get<unsigned>("ITER");

    //         index->L_dpg = index->K / 2;
    //     }

    //     void ComponentRefineDPG::Link(Index::SimpleNeighbor *cut_graph_) {
    //         std::vector<std::mutex> locks(index->getBaseLen());

    //         // PRUNE
    //         ComponentPrune *b = new ComponentPruneDPG(index);

    // #pragma omp parallel
    //         {
    //             boost::dynamic_bitset<> flags{index->getBaseLen(), 0};
    // #pragma omp for schedule(dynamic, 100)
    //             for (unsigned n = 0; n < index->getBaseLen(); ++n) {
    //                 //std::cout << n << std::endl;

    //                 flags.reset();

    //                 b->PruneInner(n, index->L_dpg, flags, index->getFinalGraph()[n], cut_graph_);
    //             }
    //         }
    //     }

}