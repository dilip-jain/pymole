#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#define ARMA_USE_OPENMP
#define ARMA_USE_SUPERLU
#include <armadillo>

#include <gradient.h>
#include <divergence.h>
#include <laplacian.h>
#include <interpol.h>
#include <robinbc.h>
#include <mixedbc.h>
#include <addscalarbc.h>
#include <interpolCtoF.h>
#include <interpolCtoN.h>
#include <interpolFtoC.h>
#include <interpolNtoC.h>

namespace py = pybind11;
using namespace arma;

/**
 * Convert Armadillo sparse matrix to scipy-compatible format (CSC)
 * Returns a dictionary with 'data', 'indices', 'indptr', 'shape' keys
 */
template<typename T>
py::object sparse_to_scipy(const T& mat) {
    // Convert to scipy sparse format (CSC - Compressed Sparse Column)
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    
    // Get CSC matrix data from Armadillo
    const sp_mat& sp = static_cast<const sp_mat&>(mat);
     sp.sync();
    
    // Extract data, indices, and indptr
    py::array_t<double> data(sp.n_nonzero);
    py::array_t<arma::uword> indices(sp.n_nonzero);
    py::array_t<arma::uword> indptr(sp.n_cols + 1);
    
    auto data_ptr = static_cast<double*>(data.request().ptr);
    auto indices_ptr = static_cast<arma::uword*>(indices.request().ptr);
    auto indptr_ptr = static_cast<arma::uword*>(indptr.request().ptr);
    
    // Copy data from Armadillo sparse matrix
    std::copy(sp.values, sp.values + sp.n_nonzero, data_ptr);
    std::copy(sp.row_indices, sp.row_indices + sp.n_nonzero, indices_ptr);
    std::copy(sp.col_ptrs, sp.col_ptrs + sp.n_cols + 1, indptr_ptr);
    
    // Create scipy.sparse.csc_matrix
    auto csc_matrix_type = scipy_sparse.attr("csc_matrix");
    auto result = csc_matrix_type(
        py::make_tuple(data, indices, indptr),
        py::make_tuple((int)sp.n_rows, (int)sp.n_cols)
    );
    
    return result;
}

arma::vec to_arma_vector(const std::vector<double>& values) {
     arma::vec result(values.size());
     for (std::size_t index = 0; index < values.size(); ++index) {
          result(index) = values[index];
     }
     return result;
}

arma::ivec to_arma_ivec(const std::vector<int>& values) {
     arma::ivec result(values.size());
     for (std::size_t index = 0; index < values.size(); ++index) {
          result(index) = values[index];
     }
     return result;
}

arma::uvec to_arma_indices(const std::vector<arma::uword>& values) {
     arma::uvec result(values.size());
     for (std::size_t index = 0; index < values.size(); ++index) {
          result(index) = values[index];
     }
     return result;
}

arma::sp_mat scipy_to_arma_sparse(py::object matrix) {
     py::object scipy_sparse = py::module_::import("scipy.sparse");
     py::object coo = scipy_sparse.attr("coo_matrix")(matrix);
     coo.attr("sum_duplicates")();
     py::tuple shape = coo.attr("shape").cast<py::tuple>();
     const arma::uword row_count = shape[0].cast<arma::uword>();
     const arma::uword column_count = shape[1].cast<arma::uword>();
     auto rows = coo.attr("row").cast<
          py::array_t<arma::uword, py::array::forcecast>>();
     auto columns = coo.attr("col").cast<
          py::array_t<arma::uword, py::array::forcecast>>();
     auto data = coo.attr("data").cast<
          py::array_t<double, py::array::forcecast>>();
     const auto row_values = rows.unchecked<1>();
     const auto column_values = columns.unchecked<1>();
     const auto data_values = data.unchecked<1>();
     arma::umat locations(2, data_values.shape(0));
     arma::vec values(data_values.shape(0));
     for (py::ssize_t index = 0; index < data_values.shape(0); ++index) {
          locations(0, index) = row_values(index);
          locations(1, index) = column_values(index);
          values(index) = data_values(index);
     }
     return arma::sp_mat(locations, values, row_count, column_count, true, true);
}

py::tuple create_scalar_bc_lhs_native(
     arma::u16 k,
     const std::vector<arma::u32>& dimensions,
     const std::vector<double>& spacings,
     const std::vector<double>& dc_values,
     const std::vector<double>& nc_values) {
     if (dimensions.empty() || dimensions.size() > 3 ||
          spacings.size() != dimensions.size() || dc_values.size() != 2 * dimensions.size() ||
          nc_values.size() != 2 * dimensions.size()) {
          throw py::value_error("Grid and boundary data must match for 1D, 2D, or 3D");
     }
     const arma::vec dc = to_arma_vector(dc_values);
     const arma::vec nc = to_arma_vector(nc_values);
     std::vector<arma::sp_mat> side_matrices;
     if (dimensions.size() == 1) {
          arma::sp_mat left, right;
          AddScalarBC::addScalarBClhs(
               k, dimensions[0], spacings[0], dc, nc, left, right);
          side_matrices = {left, right};
     } else if (dimensions.size() == 2) {
          arma::sp_mat left, right, bottom, top;
          AddScalarBC::addScalarBClhs(
               k, dimensions[0], spacings[0], dimensions[1], spacings[1],
               dc, nc, left, right, bottom, top);
          side_matrices = {left, right, bottom, top};
     } else {
          arma::sp_mat left, right, bottom, top, front, back;
          AddScalarBC::addScalarBClhs(
               k, dimensions[0], spacings[0], dimensions[1], spacings[1],
               dimensions[2], spacings[2], dc, nc,
               left, right, bottom, top, front, back);
          side_matrices = {left, right, bottom, top, front, back};
     }

     py::tuple result(side_matrices.size());
     for (std::size_t side = 0; side < side_matrices.size(); ++side) {
          result[side] = sparse_to_scipy<arma::sp_mat>(side_matrices[side]);
     }
     return result;
}

py::array_t<double> apply_scalar_bc_rhs_native(
     py::array rhs_object,
     const std::vector<double>& dc_values,
     const std::vector<double>& nc_values,
     const std::vector<std::vector<double>>& boundary_values,
     const std::vector<std::vector<arma::uword>>& side_rows) {
     if (dc_values.size() != nc_values.size() ||
          dc_values.size() != boundary_values.size() ||
          dc_values.size() != side_rows.size() ||
          (dc_values.size() != 2 && dc_values.size() != 4 && dc_values.size() != 6)) {
          throw py::value_error("Boundary data must contain two entries per grid axis");
     }
     py::array_t<double, py::array::c_style | py::array::forcecast> rhs_array(rhs_object);
     if (rhs_array.ndim() != 1) {
          throw py::value_error("rhs must be one-dimensional");
     }
     arma::vec rhs(rhs_array.shape(0));
     auto rhs_values = rhs_array.unchecked<1>();
     for (arma::uword index = 0; index < rhs.n_elem; ++index) {
          rhs(index) = rhs_values(index);
     }

     const std::size_t axis_count = dc_values.size() / 2;
     if (axis_count == 1) {
          if (boundary_values[0].size() != 1 || boundary_values[1].size() != 1) {
               throw py::value_error("1D boundary values must contain left and right scalars");
          }
          for (std::size_t side = 0; side < 2; ++side) {
               const arma::vec values = to_arma_vector({boundary_values[side][0]});
               AddScalarBC::addScalarBCrhs(
                    rhs, values, to_arma_indices(side_rows[side]));
          }
     } else {
          const arma::vec dc = to_arma_vector(dc_values);
          const arma::vec nc = to_arma_vector(nc_values);
          std::vector<arma::vec> values;
          for (const auto& side : boundary_values) {
               values.push_back(to_arma_vector(side));
          }
          if (axis_count == 2) {
               AddScalarBC::addScalarBCrhs(
                    rhs, dc, nc, values,
                    to_arma_indices(side_rows[0]), to_arma_indices(side_rows[1]),
                    to_arma_indices(side_rows[2]), to_arma_indices(side_rows[3]));
          } else {
               AddScalarBC::addScalarBCrhs(
                    rhs, dc, nc, values,
                    to_arma_indices(side_rows[0]), to_arma_indices(side_rows[1]),
                    to_arma_indices(side_rows[2]), to_arma_indices(side_rows[3]),
                    to_arma_indices(side_rows[4]), to_arma_indices(side_rows[5]));
          }
     }

     py::array_t<double> result(rhs.n_elem);
     std::copy(rhs.begin(), rhs.end(), result.mutable_data());
     return result;
}

py::array_t<double> apply_scalar_bc_rhs_1d_native(
     py::array rhs_object,
     const std::vector<double>& boundary_values,
     const std::vector<arma::uword>& indices) {
     py::array_t<double, py::array::c_style | py::array::forcecast> rhs_array(rhs_object);
     if (rhs_array.ndim() != 1) {
          throw py::value_error("rhs must be one-dimensional");
     }
     arma::vec rhs(rhs_array.shape(0));
     auto rhs_values = rhs_array.unchecked<1>();
     for (arma::uword index = 0; index < rhs.n_elem; ++index) {
          rhs(index) = rhs_values(index);
     }
     AddScalarBC::addScalarBCrhs(
          rhs, to_arma_vector(boundary_values), to_arma_indices(indices));
     py::array_t<double> result(rhs.n_elem);
     std::copy(rhs.begin(), rhs.end(), result.mutable_data());
     return result;
}

py::tuple add_scalar_bc_native(
     py::object matrix_object,
     py::array rhs_object,
     arma::u16 k,
     const std::vector<arma::u32>& dimensions,
     const std::vector<double>& spacings,
     const std::vector<double>& dc_values,
     const std::vector<double>& nc_values,
     const std::vector<std::vector<double>>& boundary_values) {
     if (dimensions.empty() || dimensions.size() > 3 ||
          spacings.size() != dimensions.size()) {
          throw py::value_error("Grid dimensions and spacings must match for 1D, 2D, or 3D");
     }
     const std::size_t axis_count = dimensions.size();
     if (dc_values.size() != 2 * axis_count || nc_values.size() != 2 * axis_count ||
          boundary_values.size() != 2 * axis_count) {
          throw py::value_error("Boundary coefficients and values must match the grid dimensions");
     }

     std::vector<bool> active_axes(axis_count, false);
     arma::uword expected_size = 1;
     for (std::size_t axis = 0; axis < axis_count; ++axis) {
          active_axes[axis] = dc_values[2 * axis] != 0.0 || dc_values[2 * axis + 1] != 0.0 ||
                                   nc_values[2 * axis] != 0.0 || nc_values[2 * axis + 1] != 0.0;
          expected_size *= dimensions[axis] + (active_axes[axis] ? 2 : 0);
     }

     arma::sp_mat system_matrix = scipy_to_arma_sparse(matrix_object);
     py::array_t<double, py::array::c_style | py::array::forcecast> rhs_array(rhs_object);
     if (rhs_array.ndim() != 1 || rhs_array.shape(0) != static_cast<py::ssize_t>(expected_size) ||
          system_matrix.n_rows != expected_size || system_matrix.n_cols != expected_size) {
          throw py::value_error("Matrix and RHS shapes do not match the selected grid and BCs");
     }
     arma::vec rhs(expected_size);
     auto rhs_values = rhs_array.unchecked<1>();
     for (arma::uword index = 0; index < expected_size; ++index) {
          rhs(index) = rhs_values(index);
     }

     arma::vec dc = to_arma_vector(dc_values);
     arma::vec nc = to_arma_vector(nc_values);
     if (axis_count == 1) {
          if (boundary_values[0].size() != 1 || boundary_values[1].size() != 1) {
               throw py::value_error("1D boundary values must contain left and right scalars");
          }
          AddScalarBC::BC1D bc;
          bc.dc = dc;
          bc.nc = nc;
          bc.v = to_arma_vector({boundary_values[0][0], boundary_values[1][0]});
          AddScalarBC::addScalarBC(system_matrix, rhs, k, dimensions[0], spacings[0], bc);
     } else if (axis_count == 2) {
          AddScalarBC::BC2D bc;
          bc.dc = dc;
          bc.nc = nc;
          for (std::size_t side = 0; side < boundary_values.size(); ++side) {
               bc.v[side] = to_arma_vector(boundary_values[side]);
          }
          AddScalarBC::addScalarBC(
               system_matrix, rhs, k, dimensions[0], spacings[0],
               dimensions[1], spacings[1], bc);
     } else {
          AddScalarBC::BC3D bc;
          bc.dc = dc;
          bc.nc = nc;
          for (std::size_t side = 0; side < boundary_values.size(); ++side) {
               bc.v[side] = to_arma_vector(boundary_values[side]);
          }
          AddScalarBC::addScalarBC(
               system_matrix, rhs, k, dimensions[0], spacings[0],
               dimensions[1], spacings[1], dimensions[2], spacings[2], bc);
     }

     py::array_t<double> result_rhs(expected_size);
     std::copy(rhs.begin(), rhs.end(), result_rhs.mutable_data());
     return py::make_tuple(sparse_to_scipy<arma::sp_mat>(system_matrix), result_rhs);
}

/**
 * Apply operator to a numpy array (dense vector)
 */
template<typename T>
py::array_t<double> apply_operator(const T& op, py::array_t<double> v) {
    auto buf = v.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<ssize_t> shape = buf.shape;
    
    // Convert numpy array to Armadillo vector
    arma::vec arma_v(ptr, shape[0], false);
    
    // Apply operator (matrix-vector multiplication)
    arma::vec result = static_cast<const sp_mat&>(op) * arma_v;
    
     py::array_t<double> output(result.n_elem);
     std::copy(result.begin(), result.end(), output.mutable_data());
     return output;
}

template<typename T>
void bind_sparse_matrix_methods(py::class_<T>& type) {
     type.def("to_scipy_sparse", &sparse_to_scipy<T>)
         .def("apply", &apply_operator<T>, py::arg("v"))
         .def("__matmul__", &apply_operator<T>)
         .def("shape", [](const T& matrix) {
              const sp_mat& sparse_matrix = static_cast<const sp_mat&>(matrix);
              return py::make_tuple(
                   static_cast<int>(sparse_matrix.n_rows),
                   static_cast<int>(sparse_matrix.n_cols));
         })
         .def("nnz", [](const T& matrix) {
              return static_cast<int>(
                   static_cast<const sp_mat&>(matrix).n_nonzero);
         });
}

template<typename T>
void bind_directional_interpolator(py::module_& module, const char* name) {
      py::class_<T> type(module, name);
      type.def(py::init([](arma::u16 k, arma::u32 m,
                                const std::vector<int>& dc, const std::vector<int>& nc) {
                 return T(k, m, to_arma_ivec(dc), to_arma_ivec(nc));
            }))
           .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n,
                                const std::vector<int>& dc, const std::vector<int>& nc) {
                 return T(k, m, n, to_arma_ivec(dc), to_arma_ivec(nc));
            }))
           .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n, arma::u32 o,
                                const std::vector<int>& dc, const std::vector<int>& nc) {
                 return T(k, m, n, o, to_arma_ivec(dc), to_arma_ivec(nc));
            }));
      bind_sparse_matrix_methods(type);
}

PYBIND11_MODULE(_operators, m) {
    m.doc() = "PyMOLE C++ backend - Mimetic operators library";

     m.def("add_scalar_bc", &add_scalar_bc_native,
            py::arg("matrix"), py::arg("rhs"), py::arg("k"),
            py::arg("dimensions"), py::arg("spacings"), py::arg("dc"),
            py::arg("nc"), py::arg("boundary_values"),
            "Apply MOLE scalar boundary conditions to a sparse system.");
         m.def("create_scalar_bc_lhs", &create_scalar_bc_lhs_native,
              py::arg("k"), py::arg("dimensions"), py::arg("spacings"),
              py::arg("dc"), py::arg("nc"),
              "Build MOLE scalar boundary-condition matrices for each side.");
         m.def("apply_scalar_bc_rhs", &apply_scalar_bc_rhs_native,
              py::arg("rhs"), py::arg("dc"), py::arg("nc"),
              py::arg("boundary_values"), py::arg("side_rows"),
              "Apply MOLE scalar boundary values to indexed RHS entries.");
         m.def("apply_scalar_bc_rhs_1d", &apply_scalar_bc_rhs_1d_native,
              py::arg("rhs"), py::arg("boundary_values"), py::arg("indices"),
              "Apply the MOLE 1D scalar boundary RHS overload.");
    
    // ==================== Gradient Operator ====================
     py::class_<Gradient>(m, "Gradient")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Gradient operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Gradient operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"), 
             py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "3D Gradient operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  dz: Grid spacing in z-direction")
        .def(py::init([](arma::u16 k, arma::u32 m, double dx,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Gradient(k, m, dx, to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("dc"), py::arg("nc"))
        .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n, double dx, double dy,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Gradient(k, m, n, dx, dy, to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"),
             py::arg("dy"), py::arg("dc"), py::arg("nc"))
        .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n, arma::u32 o,
                         double dx, double dy, double dz,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Gradient(k, m, n, o, dx, dy, dz,
                             to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"),
             py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("dc"), py::arg("nc"))
        
        .def("to_scipy_sparse", &sparse_to_scipy<Gradient>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Gradient>,
             py::arg("v"),
             "Apply gradient operator to a vector: result = G @ v")
        
        .def("__matmul__", &apply_operator<Gradient>,
             "Operator overloading: G @ v")
        
        .def("shape", [](const Gradient& g) {
            const sp_mat& sp = static_cast<const sp_mat&>(g);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Gradient& g) {
            const sp_mat& sp = static_cast<const sp_mat&>(g);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements")
     .def("getP", [](Gradient& g) {
             const arma::vec weights = g.getP();
             return std::vector<double>(weights.begin(), weights.end());
        });
    
    // ==================== Divergence Operator ====================
     py::class_<Divergence>(m, "Divergence")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Divergence operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Divergence operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"), 
             py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "3D Divergence operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  dz: Grid spacing in z-direction")
        .def(py::init([](arma::u16 k, arma::u32 m, double dx,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Divergence(k, m, dx, to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("dc"), py::arg("nc"))
        .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n, double dx, double dy,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Divergence(k, m, n, dx, dy, to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"),
             py::arg("dy"), py::arg("dc"), py::arg("nc"))
        .def(py::init([](arma::u16 k, arma::u32 m, arma::u32 n, arma::u32 o,
                         double dx, double dy, double dz,
                         const std::vector<int>& dc, const std::vector<int>& nc) {
             return Divergence(k, m, n, o, dx, dy, dz,
                               to_arma_ivec(dc), to_arma_ivec(nc));
        }), py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"),
             py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("dc"), py::arg("nc"))
        
        .def("to_scipy_sparse", &sparse_to_scipy<Divergence>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Divergence>,
             py::arg("v"),
             "Apply divergence operator to a vector: result = D @ v")
        
        .def("__matmul__", &apply_operator<Divergence>,
             "Operator overloading: D @ v")
        
        .def("shape", [](const Divergence& d) {
            const sp_mat& sp = static_cast<const sp_mat&>(d);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Divergence& d) {
            const sp_mat& sp = static_cast<const sp_mat&>(d);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements")
     .def("getQ", [](Divergence& d) {
             const arma::vec weights = d.getQ();
             return std::vector<double>(weights.begin(), weights.end());
        });
    
    // ==================== Laplacian Operator ====================
     py::class_<Laplacian>(m, "Laplacian")
        .def(py::init<arma::u16, arma::u32, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"),
             "1D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("dx"), py::arg("dy"),
             "2D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction")
        
        .def(py::init<arma::u16, arma::u32, arma::u32, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("n"), py::arg("o"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "3D Laplacian operator constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  dz: Grid spacing in z-direction")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Laplacian>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Laplacian>,
             py::arg("v"),
             "Apply Laplacian operator to a vector: result = L @ v")
        
        .def("__matmul__", &apply_operator<Laplacian>,
             "Operator overloading: L @ v")
        
        .def("shape", [](const Laplacian& l) {
            const sp_mat& sp = static_cast<const sp_mat&>(l);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const Laplacian& l) {
            const sp_mat& sp = static_cast<const sp_mat&>(l);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== Interpolation Operator ====================
     py::class_<Interpol>(m, "Interpol")
        .def(py::init<arma::u32, double>(),
             py::arg("m"), py::arg("c"),
             "1D Interpolation operator constructor\n"
             "Args:\n"
             "  m: Number of grid points\n"
             "  c: Weight for ends (0.0 <= c <= 1.0)")
        
        .def(py::init<arma::u32, arma::u32, double, double>(),
             py::arg("m"), py::arg("n"), py::arg("c1"), py::arg("c2"),
             "2D Interpolation operator constructor\n"
             "Args:\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  c1: Weight for ends in x-direction (0.0 <= c1 <= 1.0)\n"
             "  c2: Weight for ends in y-direction (0.0 <= c2 <= 1.0)")
        
        .def(py::init<arma::u32, arma::u32, arma::u32, double, double, double>(),
             py::arg("m"), py::arg("n"), py::arg("o"), 
             py::arg("c1"), py::arg("c2"), py::arg("c3"),
             "3D Interpolation operator constructor\n"
             "Args:\n"
             "  m: Number of grid points in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  c1: Weight for ends in x-direction (0.0 <= c1 <= 1.0)\n"
             "  c2: Weight for ends in y-direction (0.0 <= c2 <= 1.0)\n"
             "  c3: Weight for ends in z-direction (0.0 <= c3 <= 1.0)")
        
        .def("to_scipy_sparse", &sparse_to_scipy<Interpol>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<Interpol>,
             py::arg("v"),
             "Apply interpolation operator to a vector")
        
        .def("__matmul__", &apply_operator<Interpol>,
             "Operator overloading: I @ v")
        
        .def("shape", [](const Interpol& i) {
            const sp_mat& sp = static_cast<const sp_mat&>(i);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix")
        
        .def("nnz", [](const Interpol& i) {
            const sp_mat& sp = static_cast<const sp_mat&>(i);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== RobinBC Boundary Condition ====================
     py::class_<RobinBC>(m, "RobinBC")
        .def(py::init<arma::u16, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("a"), py::arg("b"),
             "1D Robin Boundary Condition constructor\n"
             "Boundary condition: a*u + b*du/dn = f\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing\n"
             "  a: Coefficient of Dirichlet component\n"
             "  b: Coefficient of Neumann component")
        
        .def(py::init<arma::u16, arma::u32, double, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("n"), 
             py::arg("dy"), py::arg("a"), py::arg("b"),
             "2D Robin Boundary Condition constructor\n"
             "Boundary condition: a*u + b*du/dn = f\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  a: Coefficient of Dirichlet component\n"
             "  b: Coefficient of Neumann component")
        
        .def(py::init<arma::u16, arma::u32, double, arma::u32, double, arma::u32, double, double, double>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("n"), py::arg("dy"), 
             py::arg("o"), py::arg("dz"), py::arg("a"), py::arg("b"),
             "3D Robin Boundary Condition constructor\n"
             "Boundary condition: a*u + b*du/dn = f\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dz: Grid spacing in z-direction\n"
             "  a: Coefficient of Dirichlet component\n"
             "  b: Coefficient of Neumann component")
        
        .def("to_scipy_sparse", &sparse_to_scipy<RobinBC>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<RobinBC>,
             py::arg("v"),
             "Apply Robin BC operator to a vector: result = R @ v")
        
        .def("__matmul__", &apply_operator<RobinBC>,
             "Operator overloading: R @ v")
        
        .def("shape", [](const RobinBC& r) {
            const sp_mat& sp = static_cast<const sp_mat&>(r);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const RobinBC& r) {
            const sp_mat& sp = static_cast<const sp_mat&>(r);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");
    
    // ==================== MixedBC Boundary Condition ====================
     py::class_<MixedBC>(m, "MixedBC")
        .def(py::init<arma::u16, arma::u32, double, const std::string&, 
                      const std::vector<double>&, const std::string&, 
                      const std::vector<double>&>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), 
             py::arg("left"), py::arg("coeffs_left"),
             py::arg("right"), py::arg("coeffs_right"),
             "1D Mixed Boundary Condition constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points\n"
             "  dx: Grid spacing\n"
             "  left: BC type at left boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_left: Coefficients for left BC\n"
             "  right: BC type at right boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_right: Coefficients for right BC")
        
        .def(py::init<arma::u16, arma::u32, double, arma::u32, double,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("n"), py::arg("dy"),
             py::arg("left"), py::arg("coeffs_left"),
             py::arg("right"), py::arg("coeffs_right"),
             py::arg("bottom"), py::arg("coeffs_bottom"),
             py::arg("top"), py::arg("coeffs_top"),
             "2D Mixed Boundary Condition constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  left: BC type at left boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_left: Coefficients for left BC\n"
             "  right: BC type at right boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_right: Coefficients for right BC\n"
             "  bottom: BC type at bottom boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_bottom: Coefficients for bottom BC\n"
             "  top: BC type at top boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_top: Coefficients for top BC")
        
        .def(py::init<arma::u16, arma::u32, double, arma::u32, double, arma::u32, double,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&,
                      const std::string&, const std::vector<double>&>(),
             py::arg("k"), py::arg("m"), py::arg("dx"), py::arg("n"), py::arg("dy"),
             py::arg("o"), py::arg("dz"),
             py::arg("left"), py::arg("coeffs_left"),
             py::arg("right"), py::arg("coeffs_right"),
             py::arg("bottom"), py::arg("coeffs_bottom"),
             py::arg("top"), py::arg("coeffs_top"),
             py::arg("front"), py::arg("coeffs_front"),
             py::arg("back"), py::arg("coeffs_back"),
             "3D Mixed Boundary Condition constructor\n"
             "Args:\n"
             "  k: Order of accuracy\n"
             "  m: Number of grid points in x-direction\n"
             "  dx: Grid spacing in x-direction\n"
             "  n: Number of grid points in y-direction\n"
             "  dy: Grid spacing in y-direction\n"
             "  o: Number of grid points in z-direction\n"
             "  dz: Grid spacing in z-direction\n"
             "  left: BC type at left boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_left: Coefficients for left BC\n"
             "  right: BC type at right boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_right: Coefficients for right BC\n"
             "  bottom: BC type at bottom boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_bottom: Coefficients for bottom BC\n"
             "  top: BC type at top boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_top: Coefficients for top BC\n"
             "  front: BC type at front boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_front: Coefficients for front BC\n"
             "  back: BC type at back boundary ('Dirichlet', 'Neumann', 'Robin')\n"
             "  coeffs_back: Coefficients for back BC")
        
        .def("to_scipy_sparse", &sparse_to_scipy<MixedBC>,
             "Convert to scipy sparse matrix (CSC format)")
        
        .def("apply", &apply_operator<MixedBC>,
             py::arg("v"),
             "Apply Mixed BC operator to a vector: result = M @ v")
        
        .def("__matmul__", &apply_operator<MixedBC>,
             "Operator overloading: M @ v")
        
        .def("shape", [](const MixedBC& m) {
            const sp_mat& sp = static_cast<const sp_mat&>(m);
            return py::make_tuple((int)sp.n_rows, (int)sp.n_cols);
        }, "Return shape of operator matrix (rows, cols)")
        
        .def("nnz", [](const MixedBC& m) {
            const sp_mat& sp = static_cast<const sp_mat&>(m);
            return (int)sp.n_nonzero;
        }, "Return number of non-zero elements");

     bind_directional_interpolator<InterpolCtoF>(m, "InterpolCtoF");
     bind_directional_interpolator<InterpolCtoN>(m, "InterpolCtoN");
     bind_directional_interpolator<InterpolFtoC>(m, "InterpolFtoC");
     bind_directional_interpolator<InterpolNtoC>(m, "InterpolNtoC");
}
