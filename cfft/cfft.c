#include <Python.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double re; double im; } cplx;

static size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    n |= n >> 32;
#endif
    return n + 1;
}

static void bit_reverse_permute(cplx* a, size_t n) {
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            cplx tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}

// Iterative in-place Cooley-Tukey radix-2 FFT
static void fft_iterative(cplx* a, size_t n) {
    bit_reverse_permute(a, n);
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double w_re = 1.0, w_im = 0.0;
            for (size_t j = 0; j < (len >> 1); ++j) {
                cplx u = a[i + j];
                cplx v = a[i + j + (len >> 1)];
                // v *= w
                double vr = v.re * w_re - v.im * w_im;
                double vi = v.re * w_im + v.im * w_re;
                a[i + j].re = u.re + vr;
                a[i + j].im = u.im + vi;
                a[i + j + (len >> 1)].re = u.re - vr;
                a[i + j + (len >> 1)].im = u.im - vi;
                // w *= wlen
                double nw_re = w_re * wlen_re - w_im * wlen_im;
                double nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

static PyObject* py_rfft_power(PyObject* self, PyObject* args) {
    PyObject* seq_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq_obj)) {
        return NULL;
    }

    PyObject* seq_fast = PySequence_Fast(seq_obj, "Input must be a sequence of floats");
    if (!seq_fast) {
        return NULL;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq_fast);
    if (n <= 0) {
        Py_DECREF(seq_fast);
        return Py_BuildValue("(O,i)", PyList_New(0), 0);
    }

    size_t n_in = (size_t)n;
    size_t n_fft = next_pow2(n_in);

    cplx* a = (cplx*)calloc(n_fft, sizeof(cplx));
    if (!a) {
        Py_DECREF(seq_fast);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject** items = PySequence_Fast_ITEMS(seq_fast);
    for (size_t i = 0; i < n_in; ++i) {
        double val = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            free(a);
            Py_DECREF(seq_fast);
            return NULL;
        }
        a[i].re = val;
        a[i].im = 0.0;
    }

    fft_iterative(a, n_fft);

    size_t half_n = (n_fft >> 1);
    PyObject* power_list = PyList_New(half_n + 1);
    if (!power_list) {
        free(a);
        Py_DECREF(seq_fast);
        return NULL;
    }
    for (size_t k = 0; k <= half_n; ++k) {
        double p = a[k].re * a[k].re + a[k].im * a[k].im;
        PyObject* v = PyFloat_FromDouble(p);
        if (!v) {
            Py_DECREF(power_list);
            free(a);
            Py_DECREF(seq_fast);
            return NULL;
        }
        PyList_SET_ITEM(power_list, (Py_ssize_t)k, v);
    }

    free(a);
    Py_DECREF(seq_fast);

    PyObject* ret = Py_BuildValue("(Oi)", power_list, (int)n_fft);
    return ret;
}

static PyMethodDef CfftMethods[] = {
    {"rfft_power", py_rfft_power, METH_VARARGS, "Compute RFFT power spectrum with zero-padding to next power-of-two. Returns (power_list, n_fft)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cfftmodule = {
    PyModuleDef_HEAD_INIT,
    "cfft",
    "Simple C FFT module (radix-2 Cooley-Tukey).",
    -1,
    CfftMethods
};

PyMODINIT_FUNC PyInit_cfft(void) {
    return PyModule_Create(&cfftmodule);
}


