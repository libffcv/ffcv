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
#ifdef _WIN32
    typedef unsigned __int32 __uint32_t;
    typedef unsigned __int64 __uint64_t;
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

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

    EXPORT void resize(int64_t cresizer, int64_t source_p, int64_t sx, int64_t sy,
                int64_t start_row, int64_t end_row, int64_t start_col, int64_t end_col,
                int64_t dest_p, int64_t tx, int64_t ty) {
        // TODO use proper arguments type
        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row),
                   dest_matrix, dest_matrix.size(), 0, 0, cv::INTER_AREA);
    }

  
    EXPORT void rotate(float angle, int64_t source_p, int64_t dest_p, int64_t sx, int64_t sy) {
        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(sx, sy, CV_8UC3, (uint8_t*) dest_p);
        // TODO unsure if this should be sx, sy
        cv::Point2f center((sy-1) / 2.0, (sx-1) / 2.0);
        cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(source_matrix.colRange(0, sy).rowRange(0, sx),
                   dest_matrix, rotation, dest_matrix.size(), cv::INTER_NEAREST);
    }

    EXPORT void shear(float shear_x, float shear_y, int64_t source_p, int64_t dest_p, int64_t sx, int64_t sy) {
        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(sx, sy, CV_8UC3, (uint8_t*) dest_p);
        
        float _shear[6] = { 1, shear_x, 0, shear_y, 1, 0 };
        
        float cx = (sx - 1) / 2.0;
        float cy = (sy - 1) / 2.0;

        _shear[2] += _shear[0] * -cx + _shear[1] * -cy;
        _shear[5] += _shear[3] * -cx + _shear[4] * -cy;

        _shear[2] += cx;
        _shear[5] += cy;
        
        cv::Mat shear = cv::Mat(2, 3, CV_32F, _shear);
        cv::warpAffine(source_matrix.colRange(0, sy).rowRange(0, sx),
                   dest_matrix, shear, dest_matrix.size(), cv::INTER_NEAREST);
    }
    
    EXPORT void add_weighted(int64_t img1_p, float a, int64_t img2_p, float b, int64_t dest_p, int64_t sx, int64_t sy) {
        cv::Mat img1(sx, sy, CV_8UC3, (uint8_t*) img1_p);
        cv::Mat img2(sx, sy, CV_8UC3, (uint8_t*) img2_p);
        cv::Mat dest_matrix(sx, sy, CV_8UC3, (uint8_t*) dest_p);

        // TODO doubt we need colRange/rowRange stuff
        cv::addWeighted(img1.colRange(0, sy).rowRange(0, sx), a,
                   img2.colRange(0, sy).rowRange(0, sx), b,
                   0, dest_matrix);
    }
    
    EXPORT void equalize(int64_t source_p, int64_t dest_p, int64_t sx, int64_t sy) {
        cv::Mat source_matrix(sx, sy, CV_8U, (uint8_t*) source_p);
        cv::Mat dest_matrix(sx, sy, CV_8U, (uint8_t*) dest_p);
        cv::equalizeHist(source_matrix.colRange(0, sy).rowRange(0, sx),
                        dest_matrix);
    }
    
    EXPORT void unsharp_mask(int64_t source_p, int64_t dest_p, int64_t sx, int64_t sy) {
        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(sx, sy, CV_8UC3, (uint8_t*) dest_p);
        
        cv::Point anchor(-1, -1);
        
        // 3x3 kernel, all 1s with 5 in center / sum of kernel
        float _kernel[9] = { 0.0769, 0.0769, 0.0769, 0.0769, 0.3846,
                             0.0769, 0.0769, 0.0769, 0.0769 };
        cv::Mat kernel = cv::Mat(3, 3, CV_32F, _kernel);

        cv::filter2D(source_matrix.colRange(0, sy).rowRange(0, sx),
                        dest_matrix, -1, kernel, anchor, 0, cv::BORDER_ISOLATED);

        //add_weighted(source_p, amount, dest_p, 1 - amount, dest_p, sx, sy);
    }

    EXPORT void my_memcpy(void *source, void* dst, uint64_t size) {
        memcpy(dst, source, size);
    }

    EXPORT void my_fread(int64_t fp, int64_t offset, void *destination, int64_t size) {
        fseek((FILE *) fp, offset, SEEK_SET);
        fread(destination, 1, size, (FILE *) fp);
    }

    EXPORT int imdecode(unsigned char *input_buffer, __uint64_t input_size,
                      __uint32_t source_height, __uint32_t source_width,

                      unsigned char *output_buffer,
                      __uint32_t crop_height, __uint32_t crop_width,
                      __uint32_t offset_x, __uint32_t offset_y,
                      __uint32_t scale_num, __uint32_t scale_denom,
                      bool enable_crop,
                      bool hflip)
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
        int result =  tjDecompress2(tj_decompressor, dstBuf, dstSize, output_buffer,
                TJSCALED(crop_width, scaling), 0, TJSCALED(crop_height, scaling),
                TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);

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
