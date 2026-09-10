#pragma once

#include <cstddef>
#include <vector>


class Grid {
private:
    std::size_t rows_;
    std::size_t cols_;
    // vector RAII
    std::vector<double> data_;

public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0){};


    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }
    double  operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] size_t row_size() const {return rows_;};
    [[nodiscard]] size_t col_size() const {return cols_;};
};

/// stencil logic

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    const size_t rows = old_grid.row_size();
    const size_t cols = old_grid.col_size();


    for (size_t r = 1; r < rows-1; r++) {
        for (size_t c = 1; c < cols-1; c++) {
            new_grid(r,c)= 0.5* old_grid(r, c)+
                    0.125*(old_grid(r-1,c) + old_grid(r+1,c)
                    + old_grid(r,c+1) + old_grid(r,c-1));
        }
    }

    /// edges
    for (size_t c = 0; c < cols; c++) {
        new_grid(0, c) = old_grid(0, c);
        new_grid(rows - 1, c) = old_grid(rows - 1, c);
    }
    for (size_t r = 0; r < rows; r++) {
        new_grid(r, 0) = old_grid(r, 0);
        new_grid(r, cols - 1) = old_grid(r, cols - 1);
    }

}

/// rewrite to vector
