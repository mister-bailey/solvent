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

PyDoc_STRVAR(molecule_pipeline_ext_newBatchGenerator_doc, "newBatchGenerator(batch_size, max_radius, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap)");
PyObject* molecule_pipeline_ext_newBatchGenerator(PyObject* self, PyObject* args, PyObject* kwargs) {
	int batch_size, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap;
	float max_radius;
	/* Parse positional and keyword arguments */
	static char* keywords[] = {"batch_size", "max_radius", "feature_size", "output_size", "num_threads", "molecule_cap", "example_cap", "batch_cap", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ifiiiiii|p", keywords, &batch_size, &max_radius, &feature_size, &output_size, &num_threads, &molecule_cap, &example_cap, &batch_cap))
		return NULL;

	return (PyObject *) PyCapsule_New(new BatchGenerator(batch_size, max_radius, feature_size * sizeof(ftype), output_size * sizeof(ftype), num_threads, molecule_cap, example_cap, batch_cap), "BatchGenerator", (PyCapsule_Destructor)delete_BatchGenerator);
}

PyDoc_STRVAR(molecule_pipeline_ext_newBatchGeneratorElements_doc, "newBatchGeneratorElements(batch_size, max_radius, elements, relevant_elements, num_threads, molecule_cap, example_cap, batch_cap)");
PyObject* molecule_pipeline_ext_newBatchGeneratorElements(PyObject* self, PyObject* args, PyObject* kwargs) {
	int batch_size, num_threads, molecule_cap, example_cap, batch_cap;
	float max_radius;
	PyArrayObject *elements, *relevant_elements;
	/* Parse positional and keyword arguments */
	static char* keywords[] = {"batch_size", "max_radius", "elements", "relevant_elements", "num_threads", "molecule_cap", "example_cap", "batch_cap", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ifOOiiii|p", keywords, &batch_size, &max_radius, &elements, &relevant_elements, &num_threads, &molecule_cap, &example_cap, &batch_cap))
		return NULL;

	elements = (PyArrayObject *)PyArray_FROM_OT((PyObject *)elements, NPY_INT64);
	vector<int> elements_vec;
	for(int i=0; i < PyArray_DIM(elements,0); i++) elements_vec.push_back(*((itype *)PyArray_GETPTR1(elements, i)));
	Py_DECREF(elements);

	relevant_elements = (PyArrayObject *)PyArray_FROM_OT((PyObject *)relevant_elements, NPY_INT64);
	vector<int> relevant_elements_vec;
	for(int i=0; i < PyArray_DIM(relevant_elements,0); i++) relevant_elements_vec.push_back(*((itype *)PyArray_GETPTR1(relevant_elements, i)));
	Py_DECREF(relevant_elements);

	return (PyObject *) PyCapsule_New(new BatchGenerator(batch_size, max_radius, elements_vec, relevant_elements_vec, num_threads, molecule_cap, example_cap, batch_cap), "BatchGenerator", (PyCapsule_Destructor)delete_BatchGenerator);
}


PyDoc_STRVAR(molecule_pipeline_ext_getNextBatch_doc, "(positions, features, output, weights, edge_indices, edge_vecs, name, n_examples) = getNextBatch(batch_generator, block = True)");
PyObject* molecule_pipeline_ext_getNextBatch(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject *bgc;
	bool block = true;
	static char* keywords[] = {"", "block", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", keywords, &bgc, &block))
		return NULL;

	//printf("Getting batch from C++ generator...\n");

	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(bgc, "BatchGenerator");
	Example *e = bg->getBatch(block);
	if(e == NULL) Py_RETURN_NONE;

	//printf("Building NumPy arrays...\n");
	//printf(" -- Feature size: %i bytes, %i floats\n", bg->feature_size, bg->feature_size / sizeof(ftype));

	npy_intp pdim[2] = {e->num_atoms, 3};
	PyArrayObject* positions = (PyArrayObject*)PyArray_SimpleNewFromData(2, pdim, NPY_FLOAT64, e->positions);
	PyArray_ENABLEFLAGS(positions, NPY_OWNDATA);
	npy_intp fdim[2] = {e->num_atoms, bg->feature_size / sizeof(ftype)};
	PyArrayObject* features = (PyArrayObject*)PyArray_SimpleNewFromData(2, fdim, NPY_FLOAT64, e->features);
	PyArray_ENABLEFLAGS(features, NPY_OWNDATA);
	npy_intp odim[2] = {e->num_atoms, bg->output_size / sizeof(ftype)};
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

	//printf("Returning batch from C++...\n");

	PyObject *r = Py_BuildValue("NNNNNNii", positions, features, output, weights, edge_indices, edge_vecs, e->ID, e->n_examples);
	e->releaseBuffers();
	delete e;
	return r;
}

PyDoc_STRVAR(molecule_pipeline_ext_batchReady_doc, "batchReady(batch_generator)");
PyObject* molecule_pipeline_ext_batchReady(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("O", bg->batchReady() ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_anyBatchComing_doc, "anyBatchComing(batch_generator)");
PyObject* molecule_pipeline_ext_anyBatchComing(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("O", bg->anyBatchComing() ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_moleculeQueueSize_doc, "moleculeQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_moleculeQueueSize(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("i", bg->moleculeQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_exampleQueueSize_doc, "exampleQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_exampleQueueSize(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("i", bg->exampleQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_batchQueueSize_doc, "batchQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_batchQueueSize(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("i", bg->batchQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_numExample_doc, "numExample(batch_generator)");
PyObject* molecule_pipeline_ext_numExample(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("i", bg->numExample());
}

PyDoc_STRVAR(molecule_pipeline_ext_numBatch_doc, "numBatch(batch_generator)");
PyObject* molecule_pipeline_ext_numBatch(PyObject* self, PyObject* bgc) {
	BatchGenerator* bg = (BatchGenerator*)PyCapsule_GetPointer(bgc, "BatchGenerator");
	return Py_BuildValue("i", bg->numBatch());
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

PyDoc_STRVAR(molecule_pipeline_ext_putMolecule_doc, "putMolecule(batch_generator, positions, features, output, weights, ID, block = True)");
PyObject* molecule_pipeline_ext_putMolecule(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	PyArrayObject *positions, *features, *output, *weights;
	int ID=-1;
	bool block = true;

	static char* keywords[] = { "", "positions", "features", "output", "weights", "ID", "block", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|ip", keywords, &capsule, &positions, &features, &output, &weights, &ID, &block))
		return NULL;
	BatchGenerator *bg = (BatchGenerator*)PyCapsule_GetPointer(capsule, "BatchGenerator");

	//First flush the deletion queue, since we have the GIL
	while(bg->deletion_queue.size() > 0) delete bg->deletion_queue.pop();

	//printf("Putting molecule ");
	//printf(name);
	//printf("...\n");

	positions = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)positions, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	features = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)features, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	output = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)output, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	weights = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)weights, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

	int num_examples, num_atoms;
	if(PyArray_NDIM(positions) == 3){
		num_examples = PyArray_DIM(positions, 0);
		num_atoms = PyArray_DIM(positions, 1);
	} else if(PyArray_NDIM(positions) == 2){
		num_examples = 1;
		num_atoms = PyArray_DIM(positions, 0);
	}

	Molecule *m = new Molecule(num_examples, num_atoms, positions, features, output, weights, ID);

	bool r = bg->putMolecule(m, block);
	if(!r) delete m;

	return Py_BuildValue("O", r ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_putMoleculeData_doc, "putMoleculeData(batch_generator, positions, elements, output, weights, ID, block = True)");
PyObject* molecule_pipeline_ext_putMoleculeData(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	PyArrayObject* positions, *elements, *output, *weights;
	int ID=-1;
	bool block = true;

	static char* keywords[] = { "", "positions", "elements", "output", "weights", "ID", "block", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|ip", keywords, &capsule, &positions, &elements, &output, &weights, &ID, &block))
		return NULL;
	BatchGenerator *bg = (BatchGenerator*)PyCapsule_GetPointer(capsule, "BatchGenerator");

	//First flush the deletion queue, since we have the GIL
	while(bg->deletion_queue.size() > 0) delete bg->deletion_queue.pop();

	positions = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)positions, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	elements = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)elements, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	output = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)output, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	weights = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)weights, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

	int num_examples, num_atoms;
	if(PyArray_NDIM(positions) == 3){
		num_examples = PyArray_DIM(positions, 0);
		num_atoms = PyArray_DIM(positions, 1);
	} else if(PyArray_NDIM(positions) == 2){
		num_examples = 1;
		num_atoms = PyArray_DIM(positions, 0);
	}

	printf("Ext E\n");

	Molecule *m = new Molecule(num_examples, num_atoms, positions, (itype *)PyArray_DATA(elements), output, weights, ID);

	printf("Ext G\n");

	bool r = bg->putMolecule(m, block);
	if(!r) delete m;

	Py_DECREF(elements);	
	return Py_BuildValue("O", r ? Py_True : Py_False);
}


/*
 * List of functions to add to molecule_pipeline_ext in exec_molecule_pipeline_ext().
 */
static PyMethodDef molecule_pipeline_ext_functions[] = { 
	{ "newBatchGenerator", (PyCFunction)molecule_pipeline_ext_newBatchGenerator, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_newBatchGenerator_doc },
	{ "newBatchGeneratorElements", (PyCFunction)molecule_pipeline_ext_newBatchGeneratorElements, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_newBatchGeneratorElements_doc },
	{ "getNextBatch", (PyCFunction)molecule_pipeline_ext_getNextBatch, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_getNextBatch_doc },
	{ "putMolecule", (PyCFunction)molecule_pipeline_ext_putMolecule, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_putMolecule_doc },
	{ "putMoleculeData", (PyCFunction)molecule_pipeline_ext_putMoleculeData, METH_VARARGS | METH_KEYWORDS, molecule_pipeline_ext_putMoleculeData_doc },
	{ "batchReady", (PyCFunction)molecule_pipeline_ext_batchReady, METH_O, molecule_pipeline_ext_batchReady_doc },
	{ "anyBatchComing", (PyCFunction)molecule_pipeline_ext_anyBatchComing, METH_O, molecule_pipeline_ext_anyBatchComing_doc },
	{ "moleculeQueueSize", (PyCFunction)molecule_pipeline_ext_moleculeQueueSize, METH_O, molecule_pipeline_ext_moleculeQueueSize_doc },
	{ "exampleQueueSize", (PyCFunction)molecule_pipeline_ext_exampleQueueSize, METH_O, molecule_pipeline_ext_exampleQueueSize_doc },
	{ "batchQueueSize", (PyCFunction)molecule_pipeline_ext_batchQueueSize, METH_O, molecule_pipeline_ext_batchQueueSize_doc },
	{ "numExample", (PyCFunction)molecule_pipeline_ext_numExample, METH_O, molecule_pipeline_ext_numExample_doc },
	{ "numBatch", (PyCFunction)molecule_pipeline_ext_numBatch, METH_O, molecule_pipeline_ext_numBatch_doc },
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