// idist.c

#include "Python.h"

// sequential search function
static PyObject *
knn_search_sequential(PyObject *self, PyObject *args)
{
	PyObject *dat;
	PyObject *query_pt;
	unsigned int K_;

	if (!PyArg_ParseTuple(args, "OOI:knn_search_sequential", &dat, &query_pt, &K_)) {
		return NULL;
	}

	char str[256];
	sprintf(str, "hello K_ = %u", K_);

	dat = PySequence_Fast(dat, "dat must be a sequence"); // NEW REFERENCE
	if (!dat) {
		return NULL;
	}

	Py_ssize_t i;
	Py_ssize_t n = PySequence_Fast_GET_SIZE(dat);
	for(i = 0; i < n; ++i) {
		PyObject *mat = PySequence_Fast_GET_ITEM(dat, i);


	}

	Py_DECREF(dat);
    return Py_BuildValue("s", str);

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

