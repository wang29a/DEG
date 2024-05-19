
#include "builder.h"
#include "component.h"
#include <set>

namespace stkq
{

    /**
     * load dataset and parameters
     * param data_emb_file *base_emb_norm.fvecs
     * param data_emb_file *base_loc_norm.fvecs
     * param query_emb_file *_query_emb_norm.fvecs
     * param query_loc_file *_query_loc_norm.fvecs
     * param ground_file *_groundtruth.ivecs
     * param parameters
     * return pointer of builder
     */

    IndexBuilder *IndexBuilder::load(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *ground_file, Parameters &parameters)
    {
        auto *a = new ComponentLoad(final_index_);
        a->LoadInner(data_emb_file, data_loc_file, query_emb_file, query_loc_file, ground_file, parameters);
        std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
        std::cout << "base data emb dim : " << final_index_->getBaseEmbDim() << std::endl;
        std::cout << "base data loc dim : " << final_index_->getBaseLocDim() << std::endl;
        std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
        std::cout << "query data emb dim : " << final_index_->getQueryEmbDim() << std::endl;
        std::cout << "query data loc dim : " << final_index_->getQueryLocDim() << std::endl;
        std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
        std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << final_index_->getParam().toString() << std::endl;
        std::cout << "=====================" << std::endl;
        return this;
    }
    // IndexBuilder *IndexBuilder::load(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *ground_file, char *partition_file, Parameters &parameters)
    // {
    //     auto *a = new ComponentLoad(final_index_);
    //     a->LoadInner(data_emb_file, data_loc_file, query_emb_file, query_loc_file, ground_file, parameters);
    //     a->load_partition(partition_file);
    //     std::cout << "base data len : " << final_index_->getBaseLen() << std::endl;
    //     std::cout << "base data dim : " << final_index_->getBaseDim() << std::endl;
    //     std::cout << "query data len : " << final_index_->getQueryLen() << std::endl;
    //     std::cout << "query data dim : " << final_index_->getQueryDim() << std::endl;
    //     std::cout << "ground truth data len : " << final_index_->getGroundLen() << std::endl;
    //     std::cout << "ground truth data dim : " << final_index_->getGroundDim() << std::endl;
    //     std::cout << "=====================" << std::endl;
    //     std::cout << final_index_->getParam().toString() << std::endl;
    //     std::cout << "=====================" << std::endl;
    //     return this;
    // }

    /**
     * build init graph
     * param type init type
     * param debug Whether to output graph index information (will have a certain impact on performance)
     * return pointer of builder
     */

    IndexBuilder *IndexBuilder::init(TYPE type, bool debug)
    {
        s = std::chrono::high_resolution_clock::now();
        ComponentInit *a = nullptr;

        if (type == INIT_NSW)
        {
            std::cout << "__INIT : NSW__" << std::endl;
            a = new ComponentInitNSW(final_index_);
        }
        else if (type == INIT_NSWV2)
        {
            std::cout << "__INIT : NSWV2__" << std::endl;
            a = new ComponentInitNSWV2(final_index_);
        }
        else if (type == INIT_HNSW)
        {
            std::cout << "__INIT : HNSW__" << std::endl;
            a = new ComponentInitHNSW(final_index_);
        }
        else if (type == INIT_GEO_RNG)
        {
            std::cout << "__INIT : GEO_RNG__" << std::endl;
            a = new ComponentInitGeoGraph(final_index_);
        }
        else if (type == INIT_RANDOM)
        {
            std::cout << "__INIT : RANDOM__" << std::endl;
            a = new ComponentInitRandom(final_index_);
        }
        else
        {
            std::cerr << "__INIT : WRONG TYPE__" << std::endl;
            exit(-1);
        }
        a->InitInner();
        e = std::chrono::high_resolution_clock::now();
        std::cout << "__INIT FINISH__" << std::endl;
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        // std::cout << "Initialization time: " << duration << " milliseconds" << std::endl;
        return this;
    }

    /**
     * build refine graph
     * @param type refine type
     * @param debug
     * @return
     */
    IndexBuilder *IndexBuilder::refine(TYPE type, bool debug)
    {
        ComponentRefine *a = nullptr;

        if (type == REFINE_NN_DESCENT)
        {
            std::cout << "__REFINE : NNDscent" << std::endl;
            a = new ComponentRefineNNDescent(final_index_);
        }
        else if (type == REFINE_NSG)
        {
            std::cout << "__REFINE : NSG__" << std::endl;
            a = new ComponentRefineNSG(final_index_);
        }
        // else if (type == REFINE_SSG) {
        //     std::cout << "__REFINE : NSSG__" << std::endl;
        //     a = new ComponentRefineSSG(final_index_);
        // }
        else
        {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->RefineInner();

        std::cout << "===================" << std::endl;
        std::cout << "__REFINE : FINISH__" << std::endl;
        std::cout << "===================" << std::endl;
        e = std::chrono::high_resolution_clock::now();
        return this;
    }

    IndexBuilder *IndexBuilder::save_graph(TYPE type, char *graph_file)
    {
        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        if (type == INDEX_NSG)
        {
            out.write((char *)&final_index_->ep_, sizeof(unsigned));
        }
        else if (type == INDEX_HNSW)
        {
            unsigned enterpoint_id = final_index_->enterpoint_->GetId();
            unsigned max_level = final_index_->max_level_;
            out.write((char *)&enterpoint_id, sizeof(unsigned));
            out.write((char *)&max_level, sizeof(unsigned));
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned node_level = final_index_->nodes_[i]->GetLevel() + 1;
                out.write((char *)&node_level, sizeof(unsigned));
                unsigned current_level_GK;
                for (unsigned j = 0; j < node_level; j++)
                {
                    current_level_GK = final_index_->nodes_[i]->GetFriends(j).size();
                    out.write((char *)&current_level_GK, sizeof(unsigned));
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id = final_index_->nodes_[i]->GetFriends(j)[k]->GetId();
                        out.write((char *)&current_level_neighbor_id, sizeof(unsigned));
                    }
                }
            }
            out.close();
            return this;
        }
        else if (type == INDEX_NSW)
        {
            for (unsigned i = 0; i < final_index_->nodes_.size(); i++)
            {
                unsigned GK = (unsigned)final_index_->nodes_[i]->GetFriends(0).size();
                unsigned node_id = final_index_->nodes_[i]->GetId();
                std::vector<unsigned> tmp;
                for (unsigned j = 0; j < GK; j++)
                {
                    tmp.push_back((unsigned)final_index_->nodes_[i]->GetFriends(0)[j]->GetId());
                }
                out.write((char *)&node_id, sizeof(unsigned));
                out.write((char *)&GK, sizeof(unsigned));
                out.write((char *)tmp.data(), GK * sizeof(unsigned));
            }
            out.close();
            return this;
        }
        else if (type == INDEX_NSWV2)
        {
            for (unsigned i = 0; i < final_index_->nodes_.size(); i++)
            {
                unsigned GK = (unsigned)final_index_->nodes_[i]->GetFriends(0).size();
                unsigned node_id = final_index_->nodes_[i]->GetId();
                std::vector<unsigned> tmp;
                for (unsigned j = 0; j < GK; j++)
                {
                    tmp.push_back((unsigned)final_index_->nodes_[i]->GetFriends(0)[j]->GetId());
                }
                out.write((char *)&node_id, sizeof(unsigned));
                out.write((char *)&GK, sizeof(unsigned));
                out.write((char *)tmp.data(), GK * sizeof(unsigned));
            }
            out.close();
            return this;
        }
        else if (type = INDEX_GEOGRAPH)
        {
            unsigned enterpoint_id = final_index_->geograph_enterpoint_->GetId();
            unsigned max_level = final_index_->max_level_;
            out.write((char *)&enterpoint_id, sizeof(unsigned));
            out.write((char *)&max_level, sizeof(unsigned));
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id = final_index_->geograph_nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned node_level = final_index_->geograph_nodes_[i]->GetLevel() + 1;
                out.write((char *)&node_level, sizeof(unsigned));
                for (unsigned j = 0; j < node_level; j++)
                {
                    unsigned neighbor_size = final_index_->geograph_nodes_[i]->GetFriends(j).size();
                    out.write((char *)&neighbor_size, sizeof(unsigned));
                    for (unsigned k = 0; k < neighbor_size; k++)
                    {
                        Index::GeoGraphNeighbor &neighbor = final_index_->geograph_nodes_[i]->GetFriends(j)[k];
                        unsigned neighbor_id = neighbor.id_;
                        out.write((char *)&neighbor_id, sizeof(unsigned));
                        float e_dist = neighbor.emb_distance_;
                        float s_dist = neighbor.geo_distance_;
                        int layer = neighbor.layer_;
                        out.write((char *)&e_dist, sizeof(float));
                        out.write((char *)&s_dist, sizeof(float));
                        out.write((char *)&layer, sizeof(int));
                        std::vector<std::pair<float, float>> use_range = neighbor.available_range;
                        unsigned range_size = use_range.size();
                        out.write((char *)&range_size, sizeof(unsigned));
                        for (unsigned t = 0; t < range_size; t++)
                        {
                            out.write((char *)&use_range[t].first, sizeof(float));
                            out.write((char *)&use_range[t].second, sizeof(float));
                        }
                    }
                }
            }
            out.close();
            return this;
        }

        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned GK = (unsigned)final_index_->getFinalGraph()[i].size();
            std::vector<unsigned> tmp;
            for (unsigned j = 0; j < GK; j++)
            {
                tmp.push_back(final_index_->getFinalGraph()[i][j].id);
            }
            out.write((char *)&GK, sizeof(unsigned));
            out.write((char *)tmp.data(), GK * sizeof(unsigned));
        }
        out.close();

        // std::vector<std::vector<Index::SimpleNeighbor>>().swap(final_index_->getFinalGraph());
        return this;
    }

    IndexBuilder *IndexBuilder::load_graph(TYPE type, char *graph_file)
    {
        std::ifstream in(graph_file, std::ios::binary);
        int average_neighbor_size = 0;
        int l1_average_neighbor_size = 0;

        if (!in.is_open())
        {
            std::cerr << "load graph error: " << graph_file << std::endl;
            exit(-1);
        }
        if (type == INDEX_HNSW)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in.read((char *)&enterpoint_id, sizeof(unsigned));
            in.read((char *)&final_index_->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in.read((char *)&node_id, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                in.read((char *)&node_level, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<stkq::HNSW::HnswNode *> tmp;
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + current_level_GK;
                    }
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        // final_index_->nodes_[current_level_neighbor_id]->SetId(current_level_neighbor_id);
                        tmp.push_back(final_index_->nodes_[current_level_neighbor_id]);
                    }
                    final_index_->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;

            final_index_->enterpoint_ = final_index_->nodes_[enterpoint_id];
            return this;
        }
        else if (type == INDEX_NSW)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            while (!in.eof())
            {
                unsigned GK, node_id;
                in.read((char *)&node_id, sizeof(unsigned));
                in.read((char *)&GK, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                if (in.eof())
                    break;
                std::vector<unsigned> tmp(GK);
                in.read((char *)tmp.data(), GK * sizeof(unsigned));
                for (unsigned j = 0; j < tmp.size(); j++)
                {
                    final_index_->nodes_[tmp[j]]->SetId(tmp[j]);
                    final_index_->nodes_[node_id]->AddFriends(final_index_->nodes_[tmp[j]], false);
                }
            }
            return this;
        }
        else if (type == INDEX_NSWV2)
        {
            final_index_->nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->nodes_[i] = new stkq::HNSW::HnswNode(0, 0, 0, 0);
            }
            while (!in.eof())
            {
                unsigned GK, node_id;
                in.read((char *)&node_id, sizeof(unsigned));
                in.read((char *)&GK, sizeof(unsigned));
                final_index_->nodes_[node_id]->SetId(node_id);
                if (in.eof())
                    break;
                std::vector<unsigned> tmp(GK);
                in.read((char *)tmp.data(), GK * sizeof(unsigned));
                for (unsigned j = 0; j < tmp.size(); j++)
                {
                    final_index_->nodes_[tmp[j]]->SetId(tmp[j]);
                    final_index_->nodes_[node_id]->AddFriends(final_index_->nodes_[tmp[j]], false);
                }
            }
            return this;
        }
        else if (type == INDEX_NSG)
        {
            in.read((char *)&final_index_->ep_, sizeof(unsigned));
        }
        else if (type == INDEX_GEOGRAPH)
        {
            final_index_->geograph_nodes_.resize(final_index_->getBaseLen());
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                final_index_->geograph_nodes_[i] = new stkq::GEOGRAPH::GeoGraphNode(0, 0, 0);
            }
            unsigned enterpoint_id;
            in.read((char *)&enterpoint_id, sizeof(unsigned));
            in.read((char *)&final_index_->max_level_, sizeof(unsigned));
            for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
            {
                unsigned node_id, node_level, neighbor_size;
                in.read((char *)&node_id, sizeof(unsigned));
                in.read((char *)&node_level, sizeof(unsigned));
                final_index_->geograph_nodes_[i]->SetId(node_id);
                final_index_->geograph_nodes_[i]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in.read((char *)&neighbor_size, sizeof(unsigned));
                    if (j == 0)
                    {
                        average_neighbor_size = average_neighbor_size + neighbor_size;
                    }
                    final_index_->geograph_nodes_[i]->SetMaxM(neighbor_size);
                    std::vector<Index::GeoGraphNeighbor> neighbors;
                    neighbors.reserve(neighbor_size);
                    for (unsigned k = 0; k < neighbor_size; k++)
                    {
                        unsigned neighbor_id;
                        in.read((char *)&neighbor_id, sizeof(unsigned));
                        float e_dist, s_dist;
                        int layer;
                        in.read((char *)&e_dist, sizeof(float));
                        in.read((char *)&s_dist, sizeof(float));
                        in.read((char *)&layer, sizeof(int));
                        if (layer == 0)
                        {
                            l1_average_neighbor_size = l1_average_neighbor_size + 1;
                        }
                        unsigned range_size;
                        in.read((char *)&range_size, sizeof(unsigned));
                        std::vector<std::pair<float, float>> use_range;
                        for (unsigned t = 0; t < range_size; t++)
                        {
                            float range_start, range_end;
                            in.read((char *)&range_start, sizeof(float));
                            in.read((char *)&range_end, sizeof(float));
                            use_range.push_back(std::make_pair(range_start, range_end));
                        }
                        // neighbors.push_back(std::make_shared<Index::GeoGraphEdge>(final_index_->geograph_nodes_[neighbor_id], e_dist, s_dist, use_range));
                        neighbors.push_back(Index::GeoGraphNeighbor(neighbor_id, e_dist, s_dist, use_range, layer));
                    }
                    final_index_->geograph_nodes_[i]->SetFriends(j, neighbors);
                }
            }
            std::cout << "average_neighbor_size: " << average_neighbor_size / final_index_->getBaseLen() << std::endl;
            std::cout << "l1_average_neighbor_size: " << l1_average_neighbor_size / final_index_->getBaseLen() << std::endl;
            final_index_->geograph_enterpoint_ = final_index_->geograph_nodes_[enterpoint_id];
            return this;
        }
        while (!in.eof())
        {
            unsigned GK;
            in.read((char *)&GK, sizeof(unsigned));
            if (in.eof())
                break;
            std::vector<unsigned> tmp(GK);
            in.read((char *)tmp.data(), GK * sizeof(unsigned));
            final_index_->getLoadGraph().push_back(tmp);
        }
        return this;
    }

    /**
     * offline search
     * param entry_type
     * param route_type
     * return
     */
    IndexBuilder *IndexBuilder::search(TYPE entry_type, TYPE route_type, TYPE L_type)
    {
        std::cout << "__SEARCH__" << std::endl;

        unsigned K = 10; // 在近邻搜索中要找到的最近邻的数量

        final_index_->getParam().set<unsigned>("K_search", K); // 这行代码设置了索引的参数K_search为K的值. 这意味着在后续的搜索中, 将寻找每个查询点的10个最近邻
        // final_index_->alpha = final_index_->getParam().get<float>("alpha");

        // std::vector<Index::Neighbor> pool;
        std::vector<std::vector<unsigned>> res;
        // ENTRY
        ComponentSearchEntry *a = nullptr;
        if (entry_type == SEARCH_ENTRY_NONE)
        {
            std::cout << "__SEARCH ENTRY : NONE__" << std::endl;
            a = new ComponentSearchEntryNone(final_index_);
        }
        else if (entry_type == SEARCH_ENTRY_CENTROID)
        {
            std::cout << "__SEARCH ENTRY : CENTROID__" << std::endl;
            a = new ComponentSearchEntryCentroid(final_index_);
        }
        else
        {
            std::cerr << "__SEARCH ENTRY : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        // ROUTE
        ComponentSearchRoute *b = nullptr;
        if (route_type == ROUTER_GREEDY)
        {
            std::cout << "__ROUTER : GREEDY__" << std::endl;
            b = new ComponentSearchRouteGreedy(final_index_);
        }
        else if (route_type == ROUTER_NSW)
        {
            std::cout << "__ROUTER : NSW__" << std::endl;
            b = new ComponentSearchRouteNSW(final_index_);
        }
        else if (route_type == ROUTER_HNSW)
        {
            std::cout << "__ROUTER : HNSW__" << std::endl;
            b = new ComponentSearchRouteHNSW(final_index_);
        }
        else if (route_type == ROUTER_GEOGRAPH)
        {
            std::cout << "__ROUTER : GEOGRAPH__" << std::endl;
            b = new ComponentSearchRouteGeoGraph(final_index_);
        }
        else
        {
            std::cerr << "__ROUTER : WRONG TYPE__" << std::endl;
            exit(-1);
        }
        // std::cout << final_index_->alpha << std::endl;

        if (L_type == L_SEARCH_SET_RECALL)
        {
            std::set<unsigned> visited;
            unsigned sg = 1000;
            float acc_set = 0.99;
            bool flag = false;
            int L_sl = 1;
            unsigned L = K;
            visited.insert(L);
            unsigned L_min = 0x7fffffff;
            while (true)
            {
                std::cout << "SEARCH_L : " << L << std::endl;
                if (L < K)
                {
                    std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                    exit(-1);
                }

                final_index_->getParam().set<unsigned>("L_search", L);

                auto s1 = std::chrono::high_resolution_clock::now();

                res.clear();
                res.resize(final_index_->getQueryLen());
                //  #pragma omp parallel for
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                //                for (unsigned i = 0; i < 1000; i++)
                {
                    std::vector<Index::Neighbor> pool;
                    a->SearchEntryInner(i, pool);
                    b->RouteInner(i, pool, res[i]);
                }

                auto e1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e1 - s1;
                std::cout << "search time: " << diff.count() / final_index_->getQueryLen() << "\n";
                //                std::cout << "search time: " << diff.count() / 1000 << "\n";

                // float speedup = (float)(index_->n_ * query_num) / (float)distcount;
                std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
                std::cout << "HopCount: " << final_index_->getHopCount() << std::endl;
                final_index_->resetDistCount();
                final_index_->resetHopCount();
                // int cnt = 0;
                float recall = 0;
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    if (res[i].size() == 0)
                        continue;
                    float tmp_recall = 0;
                    float cnt = 0;
                    for (unsigned j = 0; j < K; j++)
                    {
                        unsigned k = 0;
                        for (; k < K; k++)
                        {
                            if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                                break;
                        }
                        if (k == K)
                            cnt++;
                    }
                    tmp_recall = (float)(K - cnt) / (float)K;
                    recall = recall + tmp_recall;
                }
                // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                float acc = recall / final_index_->getQueryLen();
                std::cout << K << " NN accuracy: " << acc << std::endl;
                exit(1);

                if (acc_set - acc <= 0)
                {
                    if (L_min > L)
                        L_min = L;
                    if (L == K || L_sl == 1)
                    {
                        break;
                    }
                    else
                    {
                        if (flag == false)
                        {
                            L_sl < 0 ? L_sl-- : L_sl++;
                            flag = true;
                        }

                        L_sl /= 2;

                        if (L_sl == 0)
                        {
                            break;
                        }
                        L_sl < 0 ? L_sl : L_sl = -L_sl;
                    }
                }
                else
                {
                    if (L_min < L)
                        break;
                    L_sl = (int)(sg * (acc_set - acc));
                    if (L_sl == 0)
                        L_sl++;
                    flag = false;
                }
                L += L_sl;
                if (visited.count(L))
                {
                    break;
                }
                else
                {
                    visited.insert(L);
                }
            }
            std::cout << "L_min: " << L_min << std::endl;
        }
        else if (L_type == L_SEARCH_ASCEND)
        {
            unsigned L_st = 5;
            unsigned L_st2 = 8;
            for (unsigned i = 0; i < 10; i++)
            {
                unsigned L = L_st + L_st2;
                L_st = L_st2;
                L_st2 = L;
                std::cout << "SEARCH_L : " << L << std::endl;
                if (L < K)
                {
                    std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                    exit(-1);
                }

                final_index_->getParam().set<unsigned>("L_search", L);

                auto s1 = std::chrono::high_resolution_clock::now();

                res.clear();
                res.resize(final_index_->getBaseLen());

                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    std::vector<Index::Neighbor> pool;
                    a->SearchEntryInner(i, pool);
                    b->RouteInner(i, pool, res[i]);
                }

                auto e1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e1 - s1;
                std::cout << "search time: " << diff.count() << "\n";
                // float speedup = (float)(index_->n_ * query_num) / (float)distcount;
                std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
                std::cout << "HopCount: " << final_index_->getHopCount() << std::endl;
                final_index_->resetDistCount();
                final_index_->resetHopCount();
                // int cnt = 0;
                float recall = 0;
                for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
                {
                    if (res[i].size() == 0)
                        continue;
                    float tmp_recall = 0;
                    float cnt = 0;
                    for (unsigned j = 0; j < K; j++)
                    {
                        unsigned k = 0;
                        for (; k < K; k++)
                        {
                            if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                                break;
                        }
                        if (k == K)
                            cnt++;
                    }
                    tmp_recall = (float)(K - cnt) / (float)K;
                    recall = recall + tmp_recall;
                }
                // float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
                float acc = recall / final_index_->getQueryLen();
                std::cout << K << " NN accuracy: " << acc << std::endl;
            }
        }
        else if (L_type == L_SEARCH_ASSIGN)
        {

            unsigned L = final_index_->getParam().get<unsigned>("L_search");
            std::cout << "SEARCH_L : " << L << std::endl;
            if (L < K)
            {
                std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                exit(-1);
            }

            auto s1 = std::chrono::high_resolution_clock::now();

            res.clear();
            res.resize(final_index_->getBaseLen());

            for (unsigned i = 0; i < final_index_->getQueryLen(); i++)
            {
                // pool.clear();
                // if (i == 5070) continue; // only for hnsw search on glove-100
                std::vector<Index::Neighbor> pool;

                a->SearchEntryInner(i, pool);

                b->RouteInner(i, pool, res[i]);
            }

            auto e1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e1 - s1;
            std::cout << "search time: " << diff.count() << "\n";

            // float speedup = (float)(index_->n_ * query_num) / (float)distcount;
            std::cout << "DistCount: " << final_index_->getDistCount() << std::endl;
            std::cout << "HopCount: " << final_index_->getHopCount() << std::endl;
            final_index_->resetDistCount();
            final_index_->resetHopCount();
            int cnt = 0;
            for (unsigned i = 0; i < final_index_->getGroundLen(); i++)
            {
                if (res[i].size() == 0)
                    continue;
                for (unsigned j = 0; j < K; j++)
                {
                    unsigned k = 0;
                    for (; k < K; k++)
                    {
                        if (res[i][j] == final_index_->getGroundData()[i * final_index_->getGroundDim() + k])
                            break;
                    }
                    if (k == K)
                        cnt++;
                }
            }

            float acc = 1 - (float)cnt / (final_index_->getGroundLen() * K);
            std::cout << K << " NN accuracy: " << acc << std::endl;
        }

        e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }
    void IndexBuilder::peak_memory_footprint()
    {

        unsigned iPid = (unsigned)getpid();

        std::cout << "PID: " << iPid << std::endl;

        std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
        std::ifstream info(status_file);
        if (!info.is_open())
        {
            std::cout << "memory information open error!" << std::endl;
        }
        std::string tmp;
        while (getline(info, tmp))
        {
            if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
                std::cout << tmp << std::endl;
        }
        info.close();
    }
}