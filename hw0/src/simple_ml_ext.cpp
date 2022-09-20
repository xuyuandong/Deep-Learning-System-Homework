#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;



void c_mat_mul(float* C, const float* A, const float* B, size_t m, size_t l, size_t n) {
    for (auto i = 0; i < m; ++i) {
        for (auto j = 0; j < n; ++j) {
            C[i*n+j] = 0.f;
            for (auto k = 0; k < l; ++k) {
                C[i*n + j] += A[i*l + k] * B[k*n + j];
                //if (i == 0)
                //  fprintf(stderr, "--> (%zu %zu) %.4f += (%d %d) %.4f (%d, %d) %.4f\n", i, j, C[i*n + j], i,k, A[i*l+k], k, j, B[k*n+j]);
            }
        }
    }
}

void c_normalize_exp(float* Z, size_t rows, size_t cols) {
    for (auto i = 0; i < rows; ++i) {
        float sum = 0.f;
        for (auto j = 0; j < cols; ++j) {
            Z[i*cols + j] = exp(Z[i*cols + j]);
            sum += Z[i*cols + j];
        }
        
        for (auto j = 0; j < cols; ++j) {
            Z[i*cols + j] /= sum;
        }
    }
}

void c_transpose(float* Z, const float* X, size_t rows, size_t cols) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            Z[j*rows + i] = X[i*cols + j];
        }
    }
}

void c_sub_one_hot(float* Z, const unsigned char* y, size_t rows, size_t cols) {
    for (auto i = 0; i < rows; ++i) {
        Z[i*cols + y[i]] -= 1;
    }
}

void c_sub(float* z, const float* x, size_t n, float coeff) {
    for (auto i = 0; i < n; ++i) {
        z[i] -= coeff * x[i];
    }
}

void print_matrix(const float* X, size_t rows, size_t cols, const char* name, size_t maxrows=100) {
    fprintf(stderr, "[%s]==========\n", name);
    if (rows > maxrows)
        rows = maxrows;
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            fprintf(stderr, "%.4f ", X[i*cols + j]);
        }
        fprintf(stderr, "\n");
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t start = 0;
    std::vector<float> Z;
    Z.resize(batch*k, 0.f);
    std::vector<float> XT;
    XT.resize(batch*n, 0.f);
    std::vector<float> g;
    g.resize(n*k, 0.f);
    //print_matrix(theta, n, k, "theta");

    while (start < m) {
        size_t end = std::min(start + batch, m);
        //fprintf(stderr, "!!!start=%zu end=%zu m=%zu\n", start, end, m);
        //print_matrix(X + start*n, batch, n, "X", 1);

        c_mat_mul(Z.data(), X + start*n, theta, end - start, n, k);
        //print_matrix(Z.data(), batch, k, "Z", 10);

        c_normalize_exp(Z.data(), end - start, k);
        //print_matrix(Z.data(), batch, k, "eZ");

        c_sub_one_hot(Z.data(), y + start, end - start, k);
        //print_matrix(Z.data(), batch, k, "Z-y");

        c_transpose(XT.data(), X + start*n, end - start, n);
        //print_matrix(XT.data(), n, batch, "XT");

        c_mat_mul(g.data(), XT.data(), Z.data(), n, end - start, k);
        //print_matrix(g.data(), n, k, "g", 200);

        c_sub(theta, g.data(), n*k, lr/(end-start));
        //print_matrix(theta, n, k, "theta", 200);

        start += batch;
    }
    //fprintf(stderr, "finished one epoch\n");

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
