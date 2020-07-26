/******************************************************************
Python bindings for molecule_pipeline_ext
******************************************************************/

#include <Python.h>
#include <arrayobject.h>
#include <time.h>
#include <string>
#include "molecule_pipeline_imp.h"


void delete_BatchGenerator(PyObject* bgc) {
	delete (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
}

PyDoc_STRVAR(molecule_pipeline_ext_newBatchGenerator_doc, "newBatchGenerator(batch_size, max_radius, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap, go = True)");
PyObject* molecule_pipeline_ext_newBatchGenerator(PyObject* self, PyObject* args, PyObject* kwargs) {
	int batch_size, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap;
	float max_radius;
	bool go = true;
	/* Parse positional and keyword arguments */
	static char* keywords[] = {"batch_size", "max_radius", "feature_size", "output_size", "num_threads", "molecule_cap", "example_cap", "batch_cap", "go", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ifiiiiii|p", keywords, &batch_size, &max_radius, &feature_size, &output_size, &num_threads, &molecule_cap, &example_cap, &batch_cap, &go))
		return NULL;
	//printf("PyCpp return_model = %d\n", return_model);

	return (PyObject *) PyCapsule_New(new BatchGenerator(batch_size, max_radius, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap, go), "BatchGenerator", (PyCapsule_Destructor)delete_BatchGenerator);
}


PyDoc_STRVAR(molecule_pipeline_ext_getNextBatch_doc, "(positions, features, output, weights, edge_indices, edge_vecs, name) = getNextBatch(batch_generator, block = True)");
PyObject* molecule_pipeline_ext_getNextBatch(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject *bgc;
	bool block = true;
	static char* keywords[] = {"", "block", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", keywords, &bgc, &block))
		return NULL;

	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(bgc, "BatchGenerator");
	Example *e = bg->getBatch(block);
	if(e == NULL) Py_RETURN_NONE;

	//npy_intp dims[2] = { get<1>(batch), 2 };
	//printf("get<1>(batch) = %d\n", get<1>(batch));
	npy_intp pdim[2] = {e->num_atoms, 3};
	PyArrayObject* positions = (PyArrayObject*)PyArray_SimpleNewFromData(2, pdim, NPY_FLOAT64, e->positions);
	PyArray_ENABLEFLAGS(positions, NPY_OWNDATA);
	npy_intp fdim[2] = {e->num_atoms, bg->feature_size / 8};
	PyArrayObject* features = (PyArrayObject*)PyArray_SimpleNewFromData(2, fdim, NPY_FLOAT64, e->features);
	PyArray_ENABLEFLAGS(features, NPY_OWNDATA);
	npy_intp odim[2] = {e->num_atoms, bg->output_size / 8};
	PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNewFromData(2, odim, NPY_FLOAT64, e->output);
	PyArray_ENABLEFLAGS(output, NPY_OWNDATA);
	npy_intp wdim[1] = {e->num_atoms};
	PyArrayObject* weights = (PyArrayObject*)PyArray_SimpleNewFromData(1, wdim, NPY_FLOAT64, e->weights);
	PyArray_ENABLEFLAGS(weights, NPY_OWNDATA);

	npy_intp eidim[2] = {e->num_edges, 2};
	PyArrayObject* edge_indices = (PyArrayObject*)PyArray_SimpleNewFromData(2, eidim, NPY_INT64, e->edge_indices);
	PyArray_ENABLEFLAGS(edge_indices, NPY_OWNDATA);
	npy_intp evdim[2] = {e->num_edges, 3};
	PyArrayObject* edge_vecs = (PyArrayObject*)PyArray_SimpleNewFromData(2, evdim, NPY_FLOAT64, e->edge_vecs);
	PyArray_ENABLEFLAGS(edge_vecs, NPY_OWNDATA);

	PyObject *r = Py_BuildValue("NNNNNNs", positions, features, output, weights, edge_indices, edge_vecs, e->name.c_str());
	e->releaseBuffers();
	delete e;
	return r;
}

PyDoc_STRVAR(molecule_pipeline_ext_batchReady_doc, "batchReady(batch_generator)");
PyObject* molecule_pipeline_ext_batchReady(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("O", bg->batchReady() ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_notifyStarting_doc, "notifyStarting(batch_generator, batch_size = -1)");
PyObject* molecule_pipeline_ext_notifyStarting(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	int batch_size = -1;
	static char* keywords[] = { "", "batch_size", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", keywords, &capsule, &batch_size))
		return NULL;
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(capsule, "BatchGenerator");
	bg->notifyStarting(batch_size);
	Py_RETURN_NONE;
}

PyDoc_STRVAR(molecule_pipeline_ext_notifyFinished_doc, "notifyFinished(batch_generator)");
PyObject* molecule_pipeline_ext_notifyFinished(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	bg->notifyFinished();
	Py_RETURN_NONE;
}

PyDoc_STRVAR(molecule_pipeline_ext_putMolecule_doc, "putMolecule(batch_generator, positions, features, output, weights, name=\"\", block = True)");
PyObject* molecule_pipeline_ext_putMolecule(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	PyArrayObject* positions, *features, *output, *weights;
	const char *name = "";
	bool block = true;

	static char* keywords[] = { "", "positions", "features", "output", "weights", "name", "block", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|sp", keywords, &capsule, &positions, &features, &output, &weights, &name, &block))
		return NULL;
	BatchGenerator *bg = (BatchGenerator*)PyCapsule_GetPointer(capsule, "BatchGenerator");

	positions = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)positions, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	features = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)features, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	output = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)output, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	weights = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)weights, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

	int num_examples = PyArray_DIM(positions, 0);
	int num_atoms = PyArray_DIM(positions, 1);

	Molecule *m = new Molecule(num_examples, num_atoms, positions, features, output, weights, string(name));

	bool r = bg->putMolecule(m, block);

	return Py_BuildValue("O", r ? Py_True : Py_False);
}


/*
 * List of functions to add to molecule_pipeline_ext in exec_molecule_pipeline_ext().
 */
static PyMethodDef molecule_pipeline_ext_functions[] = {
	{ "newBatchGenerator", (PyCFunction)molecule_pipeline_ext_newBatchGenerator, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_newBatchGenerator_doc },
	{ "getNextBatch", (PyCFunction)molecule_pipeline_ext_getNextBatch, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_getNextBatch_doc },
	{ "putMolecule", (PyCFunction)molecule_pipeline_ext_putMolecule, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_putMolecule_doc },
	{ "batchReady", (PyCFunction)molecule_pipeline_ext_batchReady, METH_O, molecule_pipeline_ext_batchReady_doc },
	{ "notifyStarting", (PyCFunction)molecule_pipeline_ext_notifyStarting, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_notifyStarting_doc },
	{ "notifyFinished", (PyCFunction)molecule_pipeline_ext_notifyFinished, METH_O, molecule_pipeline_ext_notifyFinished_doc },
	{ NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize molecule_pipeline_ext. May be called multiple times, so avoid
 * using static state.
 */
int exec_molecule_pipeline_ext(PyObject *module) {
	import_array();
    PyModule_AddFunctions(module, molecule_pipeline_ext_functions);

    PyModule_AddStringConstant(module, "__author__", "Michael Bailey");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddIntConstant(module, "year", 2020);

    return 0; /* success */
}

/*
 * Documentation for molecule_pipeline_ext.
 */
PyDoc_STRVAR(molecule_pipeline_ext_doc, "The molecule_pipeline_ext module");


static PyModuleDef_Slot molecule_pipeline_ext_slots[] = {
    { Py_mod_exec, (void *)exec_molecule_pipeline_ext },
    { 0, NULL }
};

static PyModuleDef molecule_pipeline_ext_def = {
    PyModuleDef_HEAD_INIT,
    "molecule_pipeline_ext",
    molecule_pipeline_ext_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    molecule_pipeline_ext_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_molecule_pipeline_ext() {
    return PyModuleDef_Init(&molecule_pipeline_ext_def);
}