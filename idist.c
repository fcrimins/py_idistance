// idist.c

#include "Python.h"

// sequential search function
static PyObject *
knn_search_sequential(PyObject *self, PyObject *dat, PyObject *query_pt, PyObject *K_)
{

    return Py_BuildValue("s", "hello, world!");

}

// Module functions table.

static PyMethodDef
module_functions[] = {
    { "knn_search_sequential", knn_search_sequential, METH_VARARGS, "Do sequential search." },
    { NULL }
};

// This function is called to initialize the module.

void
initcIdist(void)
{
    Py_InitModule3("cIdist", module_functions, "iDistance search module written in C.");
}

