#pragma once
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <memory>


template <std::size_t Alignement, typename T>
static constexpr T* assume_aligned(T* ptr) noexcept {
    return static_cast<T*>(__builtin_assume_aligned(ptr,Alignement));
}


class Grid 
{
private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t row_width_;
    double* data_;

    static constexpr std::size_t align= 64;
    static constexpr ::size_t elements_per_vec= 64/sizeof(double);
public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols) {
        row_width_ = ((cols + elements_per_vec - 1) / elements_per_vec) * elements_per_vec;
        std::size_t total_elements = rows_ * row_width_;
        std::size_t total_bytes = total_elements * sizeof(double);

        void* ptr= std::aligned_alloc(align,total_bytes);
        data_ = new (ptr) double[total_elements]();
    };

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i*row_width_ + j];
    }
    double  operator()(std::size_t i, std::size_t j) const {
        return data_[i*row_width_ + j];
    }
    [[nodiscard]] size_t row_size() const {return rows_;};
    [[nodiscard]] size_t col_size() const {return cols_;};
    [[nodiscard]] size_t row_width() const {return row_width_;};
    /// access data
    [[nodiscard]] double* data(){return assume_aligned<align>(data_);};
    [[nodiscard]] const double* data() const {return assume_aligned<align>(data_);};
};

void copy_border(const Grid& old_grid, Grid& new_grid, const std::size_t& rows,
                        const std::size_t& cols, const std::size_t row_width) 
{

    const double* __restrict src= old_grid.data();
    double* __restrict dst= new_grid.data();

    std::memcpy(dst, src, row_width*sizeof(double));
    const std::size_t offset = row_width *(rows-1);
    std::memcpy(dst+offset, src+offset, row_width*sizeof(double));

    for (std::size_t i = 0; i < rows-1; ++i) {
        const std::size_t start= i*row_width;
        dst[start]= src[start];
        dst[start+cols-1]= src[start+cols-1];

    }

}

void apply_stencil(const Grid& old_grid, Grid& new_grid) 
{ 
    
    const std::size_t rows= old_grid.row_size();
    const std::size_t cols= old_grid.col_size();
    const size_t row_width= old_grid.row_width();

    const double* __restrict src= assume_aligned<64>(old_grid.data());
    double* __restrict dst= assume_aligned<64>(new_grid.data());

    #pragma omp parallel for schedule(static)
    for (size_t r = 1; r < rows - 1; r++) {
    const double* __restrict src_curr = src + (r * row_width);
    const double* __restrict src_prev = src + ((r - 1) * row_width);
    const double* __restrict src_next = src + ((r + 1) * row_width);
    double* __restrict dst_curr       = dst + (r * row_width);

    #pragma omp simd
    for (size_t c = 1; c < cols - 1; c++) {
        dst_curr[c] = 0.5   * src_curr[c] +
                      0.125 * (src_prev[c] + src_next[c] +
                               src_curr[c + 1] + src_curr[c - 1]);
    }
}
    /// border
   copy_border(old_grid, new_grid,rows, cols, row_width);
}
