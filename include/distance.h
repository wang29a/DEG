#ifndef STKQ_DISTANCE_H
#define STKQ_DISTANCE_H
#include <immintrin.h>
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace stkq
{

    class E_Distance
    {
    public:
        // template <typename T>
        // inline float sqr_dist(const T *d, const T *q, unsigned L) const
        // {
        //     float PORTABLE_ALIGN32 TmpRes[8] = {0};
        //     uint32_t num_blk16 = L >> 4;
        //     uint32_t l = L & 0b1111;

        //     __m256 diff, v1, v2;
        //     __m256 sum = _mm256_set1_ps(0);
        //     for (uint32_t i = 0; i < num_blk16; i++)
        //     {
        //         v1 = _mm256_loadu_ps(d);
        //         v2 = _mm256_loadu_ps(q);
        //         d += 8;
        //         q += 8;
        //         diff = _mm256_sub_ps(v1, v2);
        //         sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        //         v1 = _mm256_loadu_ps(d);
        //         v2 = _mm256_loadu_ps(q);
        //         d += 8;
        //         q += 8;
        //         diff = _mm256_sub_ps(v1, v2);
        //         sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        //     }
        //     for (uint32_t i = 0; i < l / 8; i++)
        //     {
        //         v1 = _mm256_loadu_ps(d);
        //         v2 = _mm256_loadu_ps(q);
        //         d += 8;
        //         q += 8;
        //         diff = _mm256_sub_ps(v1, v2);
        //         sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        //     }
        //     _mm256_store_ps(TmpRes, sum);

        //     float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
        //                 TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        //     for (uint32_t i = 0; i < l % 8; i++)
        //     {
        //         float tmp = (*q) - (*d);
        //         ret += tmp * tmp;
        //         d++;
        //         q++;
        //     }
        //     return ret;
        // }
        template <typename T>
        inline float sqr_dist(const T* d, const T* q, unsigned L) const {
            alignas(64) float TmpRes[16] = {0}; // AVX-512 一次 512 bit = 16 x float
            uint32_t num_blk16 = L >> 4;        // 16 个 float 一块
            uint32_t l = L & 0b1111;            // 剩余元素数

            __m512 diff, v1, v2;
            __m512 sum = _mm512_set1_ps(0.0f);  // 初始化 sum = 0

            // 每次处理 16 个 float
            for (uint32_t i = 0; i < num_blk16; i++) {
                v1 = _mm512_loadu_ps(d);
                v2 = _mm512_loadu_ps(q);
                d += 16;
                q += 16;
                diff = _mm512_sub_ps(v1, v2);
                sum = _mm512_fmadd_ps(diff, diff, sum); // sum += diff * diff
            }

            // 把 SIMD 累积结果存回数组
            _mm512_store_ps(TmpRes, sum);

            // 汇总 SIMD 累加的 16 个结果
            float ret = 0.0f;
            for (int i = 0; i < 16; i++) ret += TmpRes[i];

            // 处理剩余不足 16 个的元素
            for (uint32_t i = 0; i < l; i++) {
                float tmp = q[i] - d[i];
                ret += tmp * tmp;
            }

            return ret;
        }

        template <typename T>
        T compare(const T *a, const T *b, unsigned length) const
        {
            T emb_distance = sqr_dist(a, b, length);
            return std::sqrt(emb_distance) / max_emb_dist;
        }

        // template <typename T>
        // T compare(const T *a, const T *b, unsigned length) const
        // {
        //     T emb_distance = 0;
        //     if (length < 4)
        //     {
        //         for (int i = 0; i < length; i++)
        //         {
        //             emb_distance = emb_distance + (a[i] - b[i]) * (a[i] - b[i]);
        //         }
        //         return std::sqrt(emb_distance) / max_emb_dist;
        //     }

        //     float diff0, diff1, diff2, diff3;
        //     const T *last = a + length;
        //     const T *unroll_group = last - 3;

        //     /* Process 4 items with each loop for efficiency. */
        //     while (a < unroll_group)
        //     {
        //         diff0 = a[0] - b[0];
        //         diff1 = a[1] - b[1];
        //         diff2 = a[2] - b[2];
        //         diff3 = a[3] - b[3];
        //         emb_distance += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        //         a += 4;
        //         b += 4;
        //     }
        //     /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        //     while (a < last)
        //     {
        //         diff0 = *a++ - *b++;
        //         emb_distance += diff0 * diff0;
        //     }
        //     // 计算的是两个向量之间的欧几里得距离的平方，但没有进行开根号操作
        //     return std::sqrt(emb_distance) / max_emb_dist;
        // }

        E_Distance(float max_emb_dist) : max_emb_dist(max_emb_dist) {}

    private:
        float max_emb_dist = 0;
    };

    class S_Distance
    {
    public:
        template <typename T>
        T compare(const T *a, const T *b, unsigned length) const
        {
            T spatial_distance = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
            return std::sqrt(spatial_distance) / max_spatial_dist;
        }
        S_Distance(float max_spatial_dist) : max_spatial_dist(max_spatial_dist) {}

    private:
        float max_spatial_dist = 0;
    };
}

#endif
