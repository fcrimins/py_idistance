// idist.c

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

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
		PyObject *pyary = PySequence_Fast_GET_ITEM(dat, i);
		//PyArrayObject *mat = (PyArrayObject *)pyary;

		// http://mail.scipy.org/pipermail/numpy-discussion/2009-September/045367.html
		int nd = PyArray_NDIM(pyary);

		// http://stackoverflow.com/questions/2420317/why-does-my-hello-world-python-c-module-work-correctly-in-everything-but-idle
		//fprintf(stderr, "ndim = %d\n", ndim);

		int typenum = NPY_DOUBLE; // requires the numpy array to be constructed in python with explicit dtype='double'
		PyArray_Descr *descr = PyArray_DescrFromType(typenum);

		double **cary;
		npy_intp dims[3]; // PyArray_AsCArray is for ndim <= 3

//		if (PyArray_AsCArray(&pyary, (void *)&cary, dims, nd, descr) < 0) {
//			Py_DECREF(dat);
//			PyErr_SetString(PyExc_TypeError,  "error on getting c array");
//			return NULL;
//		}

		//sprintf(str, "hello K_/nd = %u/%d", K_, nd);





// @TODO: look at cython and typedmemoryviews...this is ridiculous




/*

		  npy_intp dims[3]; // PyArray_AsCArray is for ndim <= 3
		  int typenum;
		  int i, nd;
		  PyObject *o1;
		  double *d1;
		  PyArray_Descr *descr;

		  if (PyArg_ParseTuple(args, "O:print_a1", &o1) < 0) {
		    PyErr_SetString( PyExc_TypeError,  "bad arguments");
		    return NULL;
		  }

		  nd = PyArray_NDIM(o1);
		  typenum = NPY_DOUBLE;
		  descr = PyArray_DescrFromType(typenum);
		  if (PyArray_AsCArray(&o1, (void *)&d1, dims, nd, descr) < 0){
		    PyErr_SetString( PyExc_TypeError,  "error on getting c array");
		    return NULL;
		  }

		  printf("[%d] ", dims[0]);
		  for (i = 0; i < dims[0]; ++i){
		    printf("%.2f ", d1[i]);
		  }
		  printf("\n");

		  printf("if ( ((PyArrayObject *)o1)->data == d1): ");
		  if ( ((PyArrayObject *)o1)->data == (char *)d1){
		    printf("True\n");
		  }else{
		    printf("False\n");
		  }

		  if (PyArray_Free(o1, (void *)&d1) < 0){
		    PyErr_SetString( PyExc_TypeError,  "PyArray_Free fail");
		    return NULL;
		  }

*/
// @TODO: numpy c api

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

