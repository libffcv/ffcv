#include <cstdint>
#include <Python.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <turbojpeg.h>
#include <pthread.h>

extern "C" {
    // a key use to point to the tjtransform instance
    static pthread_key_t key_tj_transformer;
    // a key use to point to the tjdecompressor instance
    static pthread_key_t key_tj_decompressor;
    static pthread_once_t key_once = PTHREAD_ONCE_INIT;

    // will make the keys to access the tj instances
    static void make_keys()
    {
        pthread_key_create(&key_tj_decompressor, NULL);
        pthread_key_create(&key_tj_transformer, NULL);
    }

    void resize(int64_t cresizer, int64_t source_p, int64_t sx, int64_t sy,
                int64_t start_row, int64_t end_row, int64_t start_col, int64_t end_col,
                int64_t dest_p, int64_t tx, int64_t ty, bool is_rgb){
        int dtype;
        if (is_rgb) {
            dtype = CV_8UC3;
        } else {
            dtype = CV_8UC1;
        }

        cv::Mat source_matrix(sx, sy, dtype, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, dtype, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row),
                   dest_matrix, dest_matrix.size(), 0, 0, cv::INTER_AREA);
    }

    void my_memcpy(void *source, void* dst, uint64_t size) {
        memcpy(dst, source, size);
    }

    void my_fread(int64_t fp, int64_t offset, void *destination, int64_t size) {
        fseek((FILE *) fp, offset, SEEK_SET);
        fread(destination, 1, size, (FILE *) fp);
    }

    int imdecode(unsigned char *input_buffer, __uint64_t input_size,
                 __uint32_t source_height, __uint32_t source_width,
                 unsigned char *output_buffer,
                 __uint32_t crop_height, __uint32_t crop_width,
                 __uint32_t offset_x, __uint32_t offset_y,
                 __uint32_t scale_num, __uint32_t scale_denom,
                 bool enable_crop,
                 bool hflip,
                 bool is_rgb)
    {
        pthread_once(&key_once, make_keys);

        tjhandle tj_transformer;
        tjhandle tj_decompressor;
        if ((tj_transformer = pthread_getspecific(key_tj_transformer)) == NULL)
        {
            tj_transformer = tjInitTransform();
            pthread_setspecific(key_tj_transformer, tj_transformer);
        }
        if ((tj_decompressor = pthread_getspecific(key_tj_decompressor)) == NULL)
        {
            tj_decompressor = tjInitDecompress();
            pthread_setspecific(key_tj_decompressor, tj_decompressor);
        }

        tjtransform xform;
        tjscalingfactor scaling;
        memset(&xform, 0, sizeof(tjtransform));
        if (hflip) {
          xform.op = TJXOP_HFLIP;
        }
        xform.r.x = offset_x;
        xform.r.y = offset_y;
        xform.r.h = crop_height;
        xform.r.w = crop_width;
        xform.options |= TJXOPT_CROP;
        scaling.num = scale_num;
        scaling.denom = scale_denom;

        unsigned char *dstBuf = NULL;
        unsigned long dstSize = 0;

        bool do_transform = enable_crop || hflip;

        if (do_transform) {
            tjTransform(tj_transformer, input_buffer, input_size, 1, &dstBuf,
                        &dstSize, &xform, TJFLAG_FASTDCT);
        } else {
            dstBuf = input_buffer;
            dstSize = input_size;
        }

        TJPF pixel_format;

        if (is_rgb) {
            pixel_format = TJPF_RGB;
        } else {
            pixel_format = TJPF_GRAY;
        }

        int result =  tjDecompress2(tj_decompressor, dstBuf, dstSize, output_buffer,
                TJSCALED(crop_width, scaling), 0, TJSCALED(crop_height, scaling),
                pixel_format, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);

        if (do_transform) {
             tjFree(dstBuf);
        }

        return result;
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
