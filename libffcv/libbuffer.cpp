#include <Python.h>
#include <iostream>
extern "C" {

static PyObject* create_bytes(PyObject* self)
{
 
    // Create a bytes object
    PyObject* py_bytes = PyBytes_FromStringAndSize("asdfs", 2);

    return py_bytes;
}
static PyObject* py_bytes = NULL;

static PyObject* load_buffer(PyObject* self, PyObject* args){
    const char* filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    
    if (py_bytes == NULL) {
    FILE *f = fopen(filename, "rb");
    std::cout << "open file" << filename << std::endl;
    if (!f) {
        PyErr_SetString(PyExc_IOError, "Failed to open file");
        return NULL;
    }

    // Get the size of the file
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read the file into a buffer
    unsigned char* buffer = new unsigned char[size];
    size_t read = fread(buffer, 1, size, f);
    fclose(f);

    if (read != size) {
        delete[] buffer;
        PyErr_SetString(PyExc_IOError, "Failed to read file");
        return NULL;
    }

    // Convert the buffer to a Python bytes object
    py_bytes = PyBytes_FromStringAndSize((const char*) buffer, size);        
    }else{
        std::cout << "reuse file" << filename << std::endl;
    }

    return py_bytes;
}


static PyMethodDef methods[] = {
    {"create_bytes", (PyCFunction)create_bytes, METH_VARARGS, NULL},
    {"load_buffer", (PyCFunction)load_buffer, METH_VARARGS, NULL},   
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libbuffer", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_libbuffer(void) {
    return PyModule_Create(&module);
}

} // extern "C"