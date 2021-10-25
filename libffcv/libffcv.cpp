#include <cstdint>
#include <Python.h>
#include <opencv2/opencv.hpp>

extern "C" {
    void resize(int64_t cresizer, int64_t source_p, int64_t sx, int64_t sy,
                int64_t start_row, int64_t end_row, int64_t start_col, int64_t end_col,
                int64_t dest_p, int64_t tx, int64_t ty) {
        // TODO use proper arguments type

        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row), dest_matrix, dest_matrix.size());
    }

    void imdecode(int64_t source_data, int64_t source_size,
                     int64_t dst_data, int64_t height, int64_t width) {
        cv::Mat source_matrix(1, source_size, CV_8UC1, (uint8_t*) source_data);
        cv::Mat dest_matrix(height, width, CV_8UC3, (uint8_t*) dst_data);
        cv::InputArray source_input(source_matrix);
        cv::imdecode(source_input, cv::IMREAD_COLOR, &dest_matrix);
    }

    static PyMethodDef libffcvMethods[] = {
        {NULL, NULL, 0, NULL}
    };


    static struct PyModuleDef libffcvmodule = {
        PyModuleDef_HEAD_INIT,
        "libffcv",
        "This is a dummy python extension, the real code is available through ctypes",
        -1,
        libffcvMethods
    };

    PyMODINIT_FUNC PyInit__libffcv(void) {
        return PyModule_Create(&libffcvmodule);
    }
}
