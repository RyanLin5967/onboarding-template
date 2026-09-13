#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

struct Extent {
  std::size_t rows;
  std::size_t cols;
};

template <typename T>
struct GridView {
  T* cells;
  Extent extent;

  T* row(std::size_t i) const { return cells + i * extent.cols; }
};

class Grid {
private:
  Extent extent_;
  std::vector<double> cells_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : extent_{rows, cols}
    , cells_(rows * cols, 0.0)
  { }

  double& operator()(std::size_t i, std::size_t j) {
    return cells_[i * extent_.cols + j];
  }

  double  operator()(std::size_t i, std::size_t j) const {
    return cells_[i * extent_.cols + j];
  }

  GridView<const double> view() const { return {cells_.data(), extent_}; }
  GridView<double>       view()       { return {cells_.data(), extent_}; }
};

// out must not overlap the inputs
inline void apply_stencil_row(
  const double* above, const double* center, const double* below,
  double* __restrict out, std::size_t cols
) {
  out[0] = center[0];
  out[cols - 1] = center[cols - 1];

  #pragma omp simd
  for (std::size_t j{1}; j < cols - 1; ++j) {
    out[j] = 0.5 * center[j] + 0.125 * (above[j] + below[j] + center[j - 1] + center[j + 1]);
  }
}

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const GridView<const double> old_view{old_grid.view()};
  const GridView<double> new_view{new_grid.view()};
  const Extent extent{old_view.extent};

  // copy top/bottom rows
  std::copy_n(old_view.row(0), extent.cols, new_view.row(0));
  std::copy_n(old_view.row(extent.rows - 1), extent.cols, new_view.row(extent.rows - 1));

  #pragma omp parallel for schedule(static)
  for (std::size_t i{1}; i < extent.rows - 1; ++i) {
    apply_stencil_row(
      old_view.row(i - 1), old_view.row(i), old_view.row(i + 1), new_view.row(i), extent.cols
    );
  }
}
