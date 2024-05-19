#include "component.h"
namespace stkq
{
    template <typename T>
    inline void load_data(char *filename, T *&data, unsigned &num, unsigned &dim)
    {
        std::ifstream in(filename, std::ios::binary);
        // 创建一个输入文件流in，以二进制模式打开名为filename的文件
        if (!in.is_open())
        {
            std::cerr << "open file" << filename << " error" << std::endl;
            exit(-1);
        }
        // 检查文件是否成功打开。如果文件没有成功打开，向标准错误流输出错误信息并退出程序
        in.read((char *)&dim, 4);
        // 从文件中读取4个字节的数据到dim变量中。这假设文件的开始部分包含了一个4字节的整数，表示数据的维度
        in.seekg(0, std::ios::end);
        // 将文件流的位置指针移动到文件末尾，用于计算文件大小
        std::ios::pos_type ss = in.tellg();
        // 获取当前文件流的位置，即文件的总大小
        auto f_size = (size_t)ss;
        num = (unsigned)(f_size / (dim + 1) / 4);
        data = new T[num * dim];
        // 分配足够的内存来存储所有数据LoadInner

        in.seekg(0, std::ios::beg);
        // 将文件流的位置指针重新定位到文件开头，准备开始读取数据
        for (size_t i = 0; i < num; i++)
        {
            in.seekg(4, std::ios::cur);
            in.read((char *)(data + i * dim), dim * sizeof(T));
        }
        in.close();
    }

    void ComponentLoad::LoadInner(char *data_emb_file, char *data_loc_file, char *query_emb_file, char *query_loc_file, char *ground_file,
                                  Parameters &parameters)
    {
        // base_emb_data
        float *data_emb = nullptr;
        unsigned n{};
        unsigned emb_dim{};
        load_data<float>(data_emb_file, data_emb, n, emb_dim);
        index->setBaseEmbData(data_emb);
        index->setBaseLen(n);
        index->setBaseEmbDim(emb_dim);
        assert(index->getBaseEmbData() != nullptr && index->getBaseLen() != 0 && index->getBaseEmbDim() != 0);
        float *data_loc = nullptr;
        unsigned loc_n{};
        unsigned loc_dim{};
        load_data<float>(data_loc_file, data_loc, loc_n, loc_dim);
        index->setBaseLocData(data_loc);
        index->setBaseLocDim(loc_dim);
        assert(index->getBaseLocData() != nullptr && loc_n == index->getBaseLen());
        // query_emb_data
        float *query_emb = nullptr;
        unsigned query_num{};
        unsigned query_emb_dim{};
        load_data<float>(query_emb_file, query_emb, query_num, query_emb_dim);
        index->setQueryEmbData(query_emb);
        index->setQueryLen(query_num);
        index->setQueryEmbDim(query_emb_dim);
        assert(index->getQueryEmbData() != nullptr && index->getQueryLen() != 0 && index->getQueryEmbDim() != 0);
        assert(index->getBaseEmbDim() == index->getQueryEmbDim());
        float *query_loc = nullptr;
        unsigned query_loc_num{};
        unsigned query_loc_dim{};
        load_data(query_loc_file, query_loc, query_loc_num, query_loc_dim);
        index->setQueryLocData(query_loc);
        index->setQueryLocDim(query_loc_dim);
        assert(query_loc_num == index->getQueryLen() && query_loc_dim == index->getBaseLocDim());
        // ground_data
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        load_data<unsigned>(ground_file, ground_data, ground_num, ground_dim);
        index->setGroundData(ground_data);
        index->setGroundLen(ground_num);
        index->setGroundDim(ground_dim);
        assert(index->getGroundData() != nullptr && index->getGroundLen() != 0 && index->getGroundDim() != 0);
        // index->set_alpha(parameters.get<float>("alpha"));
        index->setParam(parameters);
        // index->set_Dist(emb_dim, loc_dim);
//        index->setBaseLen(1e4);
    }

    // void ComponentLoad::load_partition(char *partition_file)
    // {
    //     std::ifstream in(partition_file);
    //     if (!in.is_open())
    //     {
    //         std::cerr << "load partition error" << std::endl;
    //         exit(-1);
    //     }
    //     std::vector<std::vector<unsigned>> partitions;
    //     std::string line;
    //     while (std::getline(in, line))
    //     {
    //         std::istringstream iss(line);
    //         std::string part;
    //         // 读取块索引
    //         std::getline(iss, part, ':');
    //         unsigned blockIndex = std::stoi(part);

    //         // 确保分区向量足够大
    //         if (blockIndex >= partitions.size())
    //         {
    //             partitions.resize(blockIndex + 1);
    //         }

    //         // 读取并解析块中的ID
    //         while (std::getline(iss, part, ','))
    //         {
    //             if (!part.empty())
    //             {
    //                 partitions[blockIndex].push_back(std::stoi(part));
    //             }
    //         }
    //     }
    //     // index->setPartition(partitions);

    //     // std::cout << partitions.size() << std::endl;

    //     // for (size_t i = 0; i < index->getPartition().size(); ++i)
    //     // {
    //     //     std::cout << index->getPartition()[i].size() << std::endl;
    //     // }
    //     //        exit(1);
    // }
}