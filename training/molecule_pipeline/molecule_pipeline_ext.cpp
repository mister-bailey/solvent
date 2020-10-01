/******************************************************************
Python bindings for molecule_pipeline_ext
******************************************************************/

#include <Python.h>

#include <arrayobject.h>
#include <time.h>
#include <string>
#include <stdlib.h>
#include "molecule_pipeline_imp.h"


void delete_BatchGenerator(PyObject* capsule) {
	delete (BatchGenerator*)PyCapsule_GetPointer(capsule, "BatchGenerator");
}

map<int, pair<double, double>> *parse_affine_dict(PyObject *py_dict){
	map<int, pair<double, double>> *c_dict = new map<int, pair<double, double>>();
	PyObject *key, *value;
	Py_ssize_t ppos=0;
	//printf("Iterate through dictionary items...\n");
	while(true) {
		if(!PyDict_Next(py_dict, &ppos, &key, &value)) break;
		//printf("Parsing dict items: ");
		int e = PyLong_AsLong(key);
		//printf("%d ", e);
		if(!PySequence_Check(value)){
			printf("\nValue is not a sequence object!\n");
			exit(1);
		}
		if(PySequence_Size(value) != 2){
			printf("\nValue has %d elements!\n", PySequence_Size(value));
			exit(1);
		}
		//printf("(");
		double a = PyFloat_AsDouble(PySequence_ITEM(value, 0));
		//printf("%f, ", a);
		double b = PyFloat_AsDouble(PySequence_ITEM(value, 1));
		//printf("%f)\n", b);
		(*c_dict)[e] = {a,b};
	}
	//printf("Done iteration.\n");
	/*
	printf("Convert PyDict to PyList...");
	PyObject *py_list = PyDict_Items(py_dict);
	printf("Done.\n");
	int size = PyList_Size(py_list);
	for(int i; i < size; i++){
		int e;
		double a, b;
		printf("Parsing list item... ");
		PyArg_ParseTuple(PyList_GetItem(py_list, i), "i(dd)", &e, &a, &b);
		printf("Elt %d: (%f, %f)\n", e, a, b);
		(*c_dict)[e] = {a,b};
	}
	Py_DECREF(py_list);*/
	return c_dict;
}

PyDoc_STRVAR(molecule_pipeline_ext_newBatchGenerator_doc, "newBatchGenerator(batch_size, max_radius, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap)");
PyObject* molecule_pipeline_ext_newBatchGenerator(PyObject* self, PyObject* args, PyObject* kwargs) {
	int batch_size, feature_size, output_size, num_threads, molecule_cap, example_cap, batch_cap;
	float max_radius;
	/* Parse positional and keyword arguments */
	static char* keywords[] = {"batch_size", "max_radius", "feature_size", "output_size", "num_threads", "molecule_cap", "example_cap", "batch_cap", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ifiiiiii", keywords, &batch_size, &max_radius, &feature_size, &output_size, &num_threads, &molecule_cap, &example_cap, &batch_cap))
		return NULL;

	BatchGenerator *bg = new BatchGenerator(batch_size, max_radius, feature_size * sizeof(ftype), output_size * sizeof(ftype), num_threads, molecule_cap, example_cap, batch_cap);

	return (PyObject *) PyCapsule_New(bg, "BatchGenerator", (PyCapsule_Destructor)delete_BatchGenerator);
}

PyDoc_STRVAR(molecule_pipeline_ext_newBatchGeneratorElements_doc, "newBatchGeneratorElements(batch_size, max_radius, elements, relevant_elements, num_threads, molecule_cap, example_cap, batch_cap, affine = None)");
PyObject* molecule_pipeline_ext_newBatchGeneratorElements(PyObject* self, PyObject* args, PyObject* kwargs) {
	int batch_size, num_threads, molecule_cap, example_cap, batch_cap;
	float max_radius;
	PyArrayObject *elements, *relevant_elements;
	PyObject *affine=NULL;
	/* Parse positional and keyword arguments */
	static char* keywords[] = {"batch_size", "max_radius", "elements", "relevant_elements", "num_threads", "molecule_cap", "example_cap", "batch_cap", "affine", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ifOOiiii|O", keywords, &batch_size, &max_radius, &elements, &relevant_elements, &num_threads, &molecule_cap, &example_cap, &batch_cap, &affine))
		return NULL;

	map<int, pair<double, double>> *affine_dict = NULL;
	//printf("Building C dictionary...\n");
	if(affine != NULL && affine != Py_None) affine_dict = parse_affine_dict(affine);
	//printf("Done building C dictionary.\n");

	elements = (PyArrayObject *)PyArray_FROM_OT((PyObject *)elements, NPY_INT64);
	vector<int> elements_vec;
	for(int i=0; i < PyArray_DIM(elements,0); i++) elements_vec.push_back(*((itype *)PyArray_GETPTR1(elements, i)));
	Py_DECREF(elements);

	relevant_elements = (PyArrayObject *)PyArray_FROM_OT((PyObject *)relevant_elements, NPY_INT64);
	vector<int> relevant_elements_vec;
	for(int i=0; i < PyArray_DIM(relevant_elements,0); i++) relevant_elements_vec.push_back(*((itype *)PyArray_GETPTR1(relevant_elements, i)));
	Py_DECREF(relevant_elements);

	BatchGenerator *bg = new BatchGenerator(batch_size, max_radius, elements_vec, relevant_elements_vec, num_threads, molecule_cap, example_cap, batch_cap, affine_dict);
	
	delete affine_dict;
	return PyCapsule_New(bg, "BatchGenerator", (PyCapsule_Destructor)delete_BatchGenerator);
}


PyDoc_STRVAR(molecule_pipeline_ext_getNextBatch_doc, "(positions, features, output, weights, edge_indices, edge_vecs, name, n_examples) = getNextBatch(batch_generator, block = True)");
PyObject* molecule_pipeline_ext_getNextBatch(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	int block = true;
	static char* keywords[] = {"", "block", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", keywords, &capsule, &block))
		return NULL;

	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");

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

	//LINETRACK;
	PyObject *r = Py_BuildValue("NNNNNNii", positions, features, output, weights, edge_indices, edge_vecs, e->ID, e->n_examples);
	
	e->releaseBuffers();
	
	delete e;
	
	return r;
}

PyDoc_STRVAR(molecule_pipeline_ext_batchReady_doc, "batchReady(batch_generator)");
PyObject* molecule_pipeline_ext_batchReady(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("O", bg->batchReady() ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_anyBatchComing_doc, "anyBatchComing(batch_generator)");
PyObject* molecule_pipeline_ext_anyBatchComing(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("O", bg->anyBatchComing() ? Py_True : Py_False);
}

PyDoc_STRVAR(molecule_pipeline_ext_moleculeQueueSize_doc, "moleculeQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_moleculeQueueSize(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("i", bg->moleculeQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_exampleQueueSize_doc, "exampleQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_exampleQueueSize(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("i", bg->exampleQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_batchQueueSize_doc, "batchQueueSize(batch_generator)");
PyObject* molecule_pipeline_ext_batchQueueSize(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("i", bg->batchQueueSize());
}

PyDoc_STRVAR(molecule_pipeline_ext_numExample_doc, "numExample(batch_generator)");
PyObject* molecule_pipeline_ext_numExample(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("i", bg->numExample());
}

PyDoc_STRVAR(molecule_pipeline_ext_numBatch_doc, "numBatch(batch_generator)");
PyObject* molecule_pipeline_ext_numBatch(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	return Py_BuildValue("i", bg->numBatch());
}

PyDoc_STRVAR(molecule_pipeline_ext_notifyStarting_doc, "notifyStarting(batch_generator, batch_size = -1)");
PyObject* molecule_pipeline_ext_notifyStarting(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	int batch_size = -1;
	static char* keywords[] = { "", "batch_size", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", keywords, &capsule, &batch_size))
		return NULL;
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	bg->notifyStarting(batch_size);
	Py_RETURN_NONE;
}

PyDoc_STRVAR(molecule_pipeline_ext_notifyFinished_doc, "notifyFinished(batch_generator)");
PyObject* molecule_pipeline_ext_notifyFinished(PyObject* self, PyObject* capsule) {
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");
	bg->notifyFinished();
	Py_RETURN_NONE;
}

PyDoc_STRVAR(molecule_pipeline_ext_putMolecule_doc, "putMolecule(batch_generator, positions, features, output, weights, ID, block = True)");
PyObject* molecule_pipeline_ext_putMolecule(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* capsule;
	PyArrayObject *positions, *features, *output, *weights;
	int ID=-1;
	int block = true;
	
	static char* keywords[] = { "", "positions", "features", "output", "weights", "ID", "block", NULL};
	//static char* keywords[] = { "", "positions", "features", "output", "weights", "ID", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|ip", keywords, &capsule, &positions, &features, &output, &weights, &ID, &block))
	//if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|i", keywords, &capsule, &positions, &features, &output, &weights, &ID))
		return NULL;

	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");

	//First flush the deletion queue, since we have the GIL
	while(bg->deletion_queue.size() > 0) delete bg->deletion_queue.pop();

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
	int block = true;

	static char* keywords[] = { "", "positions", "elements", "output", "weights", "ID", "block", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|ip", keywords, &capsule, &positions, &elements, &output, &weights, &ID, &block))
		return NULL;
	BatchGenerator *bg = (BatchGenerator*) PyCapsule_GetPointer(capsule, "BatchGenerator");

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

