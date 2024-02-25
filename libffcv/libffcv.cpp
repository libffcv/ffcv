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

// #define _DEBUG
#ifdef _DEBUG
#define DBOUT std::cout // or any other ostream
#else
#define DBOUT 0 && std::cout
#endif

#include <utility> // For std::pair


std::pair<__uint32_t, __uint32_t> axis_to_image_boundaries(int a, int b, int img_boundary, int mcuBlock) {
    int img_b = img_boundary - (img_boundary % mcuBlock);
    int delta_a = a % mcuBlock;
    int rb = a + b;
    // reduce the a to align the mcu block
    if (a > img_b) {
        a = img_b;

    } else {
        a -= delta_a;
    }

    //  the b to align the mcu block
    // b = rb + (mcuBlock - rb % mcuBlock) - a;
    b += delta_a;

    if ((a + b) > img_b) {
        b = img_b - a;
    }

    return std::make_pair(a, b);
}

struct Boundaries {
    int x;
    int y;
    int h;
    int w;
};

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
                int64_t dest_p, int64_t tx, int64_t ty, int64_t interpolation) {
        // TODO use proper arguments type

        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row),
                   dest_matrix, dest_matrix.size(), 0, 0, interpolation);
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


    EXPORT int imcropresizedecode(unsigned char *input_buffer, __uint64_t input_size, 
                      unsigned char *tmp_buffer,                               
                      unsigned char *output_buffer,
                      __uint32_t tar_height, __uint32_t tar_width,
                      __uint32_t crop_height, __uint32_t crop_width,
                      __uint32_t offset_y, __uint32_t offset_x,
                      __uint32_t interpolation
                      )
    {
        pthread_once(&key_once, make_keys);

        tjhandle tj_transformer;
        tjhandle tj_decompressor;
        if ((tj_transformer = pthread_getspecific(key_tj_transformer)) == NULL)
        {
            tj_transformer = tj3Init(TJINIT_TRANSFORM);
            pthread_setspecific(key_tj_transformer, tj_transformer);
        }
        if ((tj_decompressor = pthread_getspecific(key_tj_decompressor)) == NULL)
        {
            tj_decompressor = tj3Init(TJINIT_DECOMPRESS);
            pthread_setspecific(key_tj_decompressor, tj_decompressor);
        }
        int result ;

        // get info about the cropped image
        int width, height, subsamp, colorspace;
        result = tjDecompressHeader3(tj_decompressor, input_buffer, input_size, &width, &height, &subsamp, &colorspace);
        if (result == -1) {
            const char* error_message = tjGetErrorStr();
            // Handle the error
            std::cerr << "Error in info: " << error_message << std::endl;
            return -1;
        }
        else {
            DBOUT << "width: " << width << " height: " << height << " Subsamp: " << subsamp << " Colorspace: " << colorspace << std::endl;
        }

        // get the boundaries of the cropped image
        std::pair<int, int> x_boundaries = axis_to_image_boundaries(offset_x, crop_width, width,  tjMCUWidth[subsamp]);
        std::pair<int, int> y_boundaries = axis_to_image_boundaries(offset_y, crop_height, height, tjMCUWidth[subsamp]);

        // reduce the crop size if it is out of the image boundaries
        int lbound = x_boundaries.first + x_boundaries.second;
        if(lbound<offset_x+crop_width){
            crop_width = lbound-x_boundaries.first;
        }
        lbound = y_boundaries.first + y_boundaries.second;
        if(lbound<offset_y+crop_height){
            crop_height = lbound - y_boundaries.first;
        }

        DBOUT << "offset_x: " << offset_x << ", " << crop_width << "," << width << " -> ";
        DBOUT << "x_boundaries: " << x_boundaries.first << ", " << x_boundaries.second << std::endl;
        DBOUT << offset_x + crop_width << " <= " << x_boundaries.second+x_boundaries.first <<" <= " << width << std::endl;

        DBOUT << "offset_y: " << offset_y << ", " << crop_height << ", " <<height <<" -> ";
        DBOUT << "y_boundaries: " << y_boundaries.first << ", " << y_boundaries.second << std::endl;
        DBOUT << offset_y + crop_height << " < " << y_boundaries.second+y_boundaries.first <<" <= " << height << std::endl;

        
        offset_x = x_boundaries.first;
        offset_y = y_boundaries.first;
        crop_width = x_boundaries.second;
        crop_height = y_boundaries.second;

        // if it is not possible to crop the image, return the original image
        if (crop_width<8){
            // std::cerr << "Invalid crop width " << crop_width <<std::endl;
            offset_x = 0;
            crop_width = width;
        }
        if (crop_height<8){
            // std::cerr << "Invalid crop height " << crop_height <<std::endl;
            offset_y = 0;
            crop_height = height;
        }

        tjtransform xform;
        tjscalingfactor scaling;
        memset(&xform, 0, sizeof(tjtransform));

        xform.r.x = offset_x;
        xform.r.y = offset_y;
        xform.r.w = crop_width;
        xform.r.h = crop_height;
        xform.op = TJXOP_NONE;
        xform.options = TJXOPT_CROP;
        scaling.num = 1;
        scaling.denom = 1;

        unsigned char *dstBuf = NULL;
        unsigned long dstSize = 0;

        // crop the input image
        tj3SetCroppingRegion(tj_decompressor, xform.r);
        
        result =  tj3Decompress8(tj_decompressor, input_buffer, input_size, tmp_buffer,
                 0,  TJPF_RGB);
        
        if (result == -1) {
            const char* error_message =  tj3GetErrorStr(tj_decompressor);
            // Handle the error
            std::cerr << "Error in tj3Decompress8: " << error_message << std::endl;
            return -1;
        }

        // resize the cropped image
        cv::Mat source_matrix(crop_height, crop_width, CV_8UC3, (uint8_t*) tmp_buffer);
        cv::Mat dest_matrix(tar_height, tar_width, CV_8UC3, (uint8_t*) output_buffer);

        // int dx = offset_x-x_boundaries.first;
        // int dy = offset_y-y_boundaries.first;

        cv::resize((source_matrix
            // .colRange(dx,crop_width)
            // .rowRange(dy,crop_height)
            ),
                   dest_matrix, dest_matrix.size(), 0, 0, interpolation);
        return result;
    }
}
