#pragma once
#include <cstddef>
#include <cstring>
#include <new>
#ifdef _OPENMP
#include <omp.h>
#endif

class Grid {
private:
    std::size_t rows_;
    std::size_t cols_;
    double* data_;

public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols),
          data_(static_cast<double*>(
              ::operator new[](rows * cols * sizeof(double), std::align_val_t{64})))
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t r = 0; r < rows_; ++r) {
            std::memset(data_ + r * cols_, 0, cols_ * sizeof(double));
        }
    }

    ~Grid() { ::operator delete[](data_, std::align_val_t{64}); }

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    Grid(Grid&& o) noexcept : rows_(o.rows_), cols_(o.cols_), data_(o.data_) {
        o.data_ = nullptr;
    }
    Grid& operator=(Grid&& o) noexcept {
        if (this != &o) {
            ::operator delete[](data_, std::align_val_t{64});
            rows_ = o.rows_; cols_ = o.cols_; data_ = o.data_;
            o.data_ = nullptr;
        }
        return *this;
    }

    inline double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    inline double  operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    [[nodiscard]] double* data() noexcept { return data_; }
    [[nodiscard]] const double* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t row_size() const { return rows_; }
    [[nodiscard]] std::size_t col_size() const { return cols_; }
};

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    const std::size_t rows = old_grid.row_size();
    const std::size_t cols = old_grid.col_size();

    const double* __restrict old_ptr = old_grid.data();
    double* __restrict new_ptr = new_grid.data();

    constexpr double c0 = 0.5;
    constexpr double c1 = 0.125;

    #pragma omp parallel for schedule(static)
    for (std::size_t r = 1; r < rows - 1; ++r) {
        const double* __restrict row      = old_ptr + r * cols;
        const double* __restrict row_up   = row - cols;
        const double* __restrict row_down = row + cols;
        double* __restrict out            = new_ptr + r * cols;

        #pragma omp simd
        for (std::size_t c = 1; c < cols - 1; ++c) {
            out[c] = c0 * row[c] + c1 * (row_up[c] + row_down[c] + row[c + 1] + row[c - 1]);
        }
    }

    std::memcpy(new_ptr, old_ptr, cols * sizeof(double));
    std::memcpy(new_ptr + (rows - 1) * cols, old_ptr + (rows - 1) * cols, cols * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (std::size_t r = 0; r < rows; ++r) {
        new_ptr[r * cols] = old_ptr[r * cols];
        new_ptr[r * cols + cols - 1] = old_ptr[r * cols + cols - 1];
    }
}
