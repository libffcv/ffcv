#include <cstdint>
#include <Python.h>
#include <opencv2/opencv.hpp>

extern "C" {
    void resize(int64_t cresizer, int64_t source_p, int64_t sx, int64_t sy,
                int64_t start_row, int64_t end_row, int64_t start_col, int64_t end_col,
                int64_t dest_p, int64_t tx, int64_t ty) {

        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row), dest_matrix, dest_matrix.size());
    }

    static PyMethodDef libfastercvMethods[] = {
        {NULL, NULL, 0, NULL}
    };


    static struct PyModuleDef libfastercvmodule = {
        PyModuleDef_HEAD_INIT,
        "libfastercv",
        "This is a dummy python extension, the real code is available through ctypes",
        -1,
        libfastercvMethods
    };

    PyMODINIT_FUNC PyInit_libfastercv(void) {
        return PyModule_Create(&libfastercvmodule);
    }
}