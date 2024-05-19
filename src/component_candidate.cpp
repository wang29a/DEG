#include "component.h"

namespace stkq
{

    // NO LIMIT GREEDY
    void ComponentCandidateNSG::CandidateInner(const unsigned query, const unsigned enter, boost::dynamic_bitset<> flags,
                                               std::vector<Index::SimpleNeighbor> &result)
    {
        auto L = index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids(L);
        std::vector<Index::Neighbor> retset;
        retset.resize(L + 1);

        L = 0;

        for (unsigned i = 0; i < init_ids.size() && i < index->getFinalGraph()[enter].size(); i++)
        {
            init_ids[i] = index->getFinalGraph()[enter][i].id;
            flags[init_ids[i]] = true;
            L++;
        }

        while (L < init_ids.size())
        {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> SimpleNeighbor
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen())
                continue;

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + id * index->getBaseLocDim(),
                                                     index->getBaseLocData() + query * index->getBaseLocDim(),
                                                     index->getBaseLocDim());

            float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

            retset[i] = Index::Neighbor(id, dist, true);
            result.emplace_back(Index::SimpleNeighbor(id, dist));
            L++;
        }
        std::sort(retset.begin(), retset.begin() + L);
        index->i++;

        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                for (unsigned m = 0; m < index->getFinalGraph()[n].size(); ++m)
                {

                    unsigned id = index->getFinalGraph()[n][m].id;

                    if (flags[id])
                        continue;
                    flags[id] = true;

                    float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                             index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                             index->getBaseEmbDim());

                    float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + query * index->getBaseLocDim(),
                                                             index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                             index->getBaseLocDim());

                    float dist = index->get_alpha() * e_d + (1 - index->get_alpha()) * s_d;

                    Index::Neighbor nn(id, dist, true);
                    result.push_back(Index::SimpleNeighbor(id, dist));

                    if (dist >= retset[L - 1].distance)
                        continue;

                    int r = Index::InsertIntoPool(retset.data(), L, nn);
                    if (L + 1 < retset.size())
                        ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

        std::vector<Index::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }

    void ComponentCandidateGeoGraph::CandidateInner(const unsigned query, const std::vector<unsigned> enter, boost::dynamic_bitset<> flags,
                                                    std::vector<Index::GeoGraphNNDescentNeighbor> &result)
    {
        auto L = index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids(L);
        // std::vector<Index::GeoGraphNNDescentNeighbor> retset;
        result.reserve(L + 1);

        L = 0;

        for (unsigned i = 0; i < init_ids.size() && i < enter.size(); i++)
        {
            init_ids[i] = enter[i];
            flags[init_ids[i]] = true;
            L++;
        }

        while (L < init_ids.size())
        {
            unsigned id = rand() % index->getBaseLen();
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }
        L = 0;
        // unsinged -> SimpleNeighbor
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= index->getBaseLen())
                continue;

            float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + id * index->getBaseEmbDim(),
                                                     index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                     index->getBaseEmbDim());

            float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + id * index->getBaseLocDim(),
                                                     index->getBaseLocData() + query * index->getBaseLocDim(),
                                                     index->getBaseLocDim());
            result.emplace_back(id, e_d, s_d, true, -1);
            L++;
        }
        std::sort(result.begin(), result.end());
        auto queue = Index::skyline_queue(L);
        queue.init_queue(result);

        int k = 0;
        int l = 0;
        while (k < queue.pool.size())
        {
            while (queue.pool[k].layer_ == l)
            {
                if (queue.pool[k].flag)
                {
                    queue.pool[k].flag = false;
                    unsigned n = queue.pool[k].id_;
                    for (unsigned m = 0; m < index->getSkylineGraph()[n].pool.size(); ++m)
                    {
                        unsigned id = index->getSkylineGraph()[n].pool[m].id_;
                        if (flags[id])
                            continue;
                        flags[id] = true;
                        float e_d = index->get_E_Dist()->compare(index->getBaseEmbData() + (size_t)id * index->getBaseEmbDim(),
                                                                 index->getBaseEmbData() + query * index->getBaseEmbDim(),
                                                                 index->getBaseEmbDim());

                        float s_d = index->get_S_Dist()->compare(index->getBaseLocData() + (size_t)id * index->getBaseLocDim(),
                                                                 index->getBaseLocData() + query * index->getBaseLocDim(),
                                                                 index->getBaseLocDim());
                        queue.pool.emplace_back(id, e_d, s_d, true, -1);
                    }
                }
                k++;
            }
            // int nk = k;
            // int nl = l;
            // queue.updateNeighbor(nk, nl);
            // if (nk < k)
            // {
            //     k = nk;
            //     l = nl;
            // }
            // else
            // {
            //     l++;
            // }
        }
        result.swap(queue.pool);
        std::vector<unsigned>().swap(init_ids);
    }
}