#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> cells_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_{rows}
    , cols_{cols}
    , cells_(rows * cols, 0.0)
  { }

  double& operator()(std::size_t i, std::size_t j) {
    return cells_[i * cols_ + j];
  }

  double  operator()(std::size_t i, std::size_t j) const {
    return cells_[i * cols_ + j];
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double* data() { return cells_.data(); }
  const double* data() const { return cells_.data(); }
};

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  const double* src = old_grid.data();
  double* dst = new_grid.data();

  // copy top/bottom rows
  std::copy_n(src, cols, dst);
  std::copy_n(src + (rows - 1) * cols, cols, dst + (rows - 1) * cols);

  #pragma omp parallel for schedule(static)
  for (std::size_t i{1}; i < rows - 1; ++i) {
    const double* above = src + (i - 1) * cols;
    const double* center = src + i * cols;
    const double* below = src + (i + 1) * cols;
    double* out = dst + i * cols;

    out[0] = center[0];
    out[cols - 1] = center[cols - 1];

    #pragma omp simd
    for (std::size_t j{1}; j < cols - 1; ++j) {
      out[j] = 0.5 * center[j] + 0.125 * (above[j] + below[j] + center[j - 1] + center[j + 1]);
    }
  }
}
