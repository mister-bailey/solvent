/******************************************************************
Copyright Michael Bailey 2020
******************************************************************/
#include "molecule_pipeline_imp.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

Molecule::Molecule(int num_examples, int num_atoms, PyArrayObject *positions, PyArrayObject *features,
            PyArrayObject *output, PyArrayObject * weights, string name) :
            num_examples(num_examples), num_atoms(num_atoms), positions(positions),
            features(features), output(output), weights(weights), name(name)
            {   
}

Molecule::Molecule(int num_examples, int num_atoms, PyArrayObject *positions, const itype *elements,
            PyArrayObject *output, PyArrayObject * weights, string name) :
            num_examples(num_examples), num_atoms(num_atoms), positions(positions),
            output(output), weights(weights), name(name) {
    printf("Constructing molecule from data A\n");
    npy_intp fdim[2] = {num_atoms, BatchGenerator::num_elements};
    printf("Constructing molecule from data B\n");
    features = (PyArrayObject *)PyArray_ZEROS(2, fdim, NPY_FLOAT64, 0);
    printf("Constructing molecule from data C\n");
    ftype *fdata = (ftype *)PyArray_DATA(features);
    printf("Constructing molecule from data D ");
    for(int a = 0; a < num_atoms; a++, fdata += BatchGenerator::num_elements){
        printf("%d ",a);
        int atomic_number = elements[a];
        fdata[BatchGenerator::element_map[atomic_number]] = 1;
        if(!BatchGenerator::relevant_elements.count(atomic_number)) ((ftype *)PyArray_DATA(weights))[a] = 0;
    }        
    printf("\nConstructing molecule from data - Finished\n");
}

Molecule::~Molecule(){
    Py_DECREF(positions);
    Py_DECREF(features);
    Py_DECREF(output);
    Py_DECREF(weights);
}

Example::Example(int num_atoms, vec3 *positions, void *features, void *output, ftype *weights,
            int num_edges, pair<itype,itype> *edge_indices, vec3 *edge_vecs, string name, int n_examples) :
            num_atoms(num_atoms), positions(positions), features(features), output(output), weights(weights),
            num_edges(num_edges), edge_indices(edge_indices), edge_vecs(edge_vecs), name(name), n_examples(n_examples) {
}

Example::~Example(){
    free(positions);
    free(features);
    free(output);
    free(weights);
    free(edge_indices);
    free(edge_vecs);
}

void Example::releaseBuffers(){
    positions = NULL;
    features = NULL;
    output = NULL;
    weights = NULL;
    edge_indices = NULL;
    edge_vecs = NULL;
}

int BatchGenerator::num_elements;
map<int, int> BatchGenerator::element_map;
set<int> BatchGenerator::relevant_elements;


void BatchGenerator::processMolecule(Molecule *m) {
    ftype max_r2 = max_radius * max_radius;
    int tot_fsize = feature_size * m->num_atoms;
    int tot_ysize = output_size * m->num_atoms;
    int tot_psize = sizeof(vec3) * m->num_atoms;
    int tot_wsize = sizeof(ftype) * m->num_atoms;
    vec3 *pos = data3(m->positions);
    void *feat = data(m->features);
    void *out = data(m->output);
    ftype *wt = (ftype *)data(m->weights);


    for(int i = 0; i < m->num_examples; i++){
        int num_edges = 0;
        for(int a = 0; a < m->num_atoms; a++){
            for(int b = 0; b < m->num_atoms; b++){
                vec3 d = diff(pos[b], pos[a]);
                if(norm2(d) <= max_r2) num_edges++;
            }
        }
        
        pair<itype,itype> *edge_indices;
        vec3 *edge_vecs;
        {MEMSAFE edge_indices = (pair<itype,itype> *)malloc(num_edges * sizeof(pair<itype,itype>));}
        {MEMSAFE edge_vecs = (vec3 *)malloc(num_edges * sizeof(vec3));}
        int edge=0;
        for(int a = 0; a < m->num_atoms; a++){
            for(int b = 0; b < m->num_atoms; b++){
                vec3 d = diff(pos[b], pos[a]);
                if(norm2(d) > max_r2) continue;
                edge_indices[edge] = {a,b};
                edge_vecs[edge] = d;
                edge++;
            }
        }

        Example *example;
        {MEMSAFE example = new Example(m->num_atoms, (vec3 *)malloc(tot_psize),
                malloc(tot_fsize), malloc(tot_ysize), (ftype *)malloc(tot_wsize), num_edges, edge_indices, edge_vecs, m->name);}
        memcpy(example->positions, pos, tot_psize);
        memcpy(example->features, feat, tot_fsize);
        memcpy(example->output, out, tot_ysize);
        memcpy(example->weights, wt, tot_wsize);

        example_queue.push(example);
        pos = (vec3 *)((char *)pos + tot_psize);
        //feat = (char *)feat + tot_fsize;
        out = (char *)out + tot_ysize;
    }
    deletion_queue.push(m);
}

void BatchGenerator::loopProcessMolecules(int max){
    for(int i=0; i != max; i++){
        //printf("m.pop\n");
        Molecule *m = molecule_queue.pop();
        //printf("m.process\n");
        processMolecule(m);
    }
}

bool BatchGenerator::putMolecule(Molecule *m, bool block){
    if(m == NULL) return false;
    {
        unique_lock<mutex> lock(ex_count_mutex);
        num_ex += m->num_examples;
        knows_ex_coming = true;
        know_cond.notify_all();
    }
    return molecule_queue.push(m, block);
}

Example *BatchGenerator::makeBatch() {
    vector<Example *> examples;
    examples.reserve(batch_size);
    int total_atoms = 0;
    int total_edges = 0;
    while(examples.size() < batch_size && anyExComing()){
        Example *e = example_queue.pop();
        if(e == NULL) break;
        {
            unique_lock<mutex> lock(ex_count_mutex);
            num_ex--;
            if(num_ex == 0 && !finished_reading) knows_ex_coming = false;
        }
        examples.push_back(e);
        total_atoms += e->num_atoms;
        total_edges += e->num_edges;
    }
    if(examples.size()==0) return NULL;
    string name = examples[0]->name;

    vec3 *positions;
    {MEMSAFE positions = (vec3 *)malloc(total_atoms * sizeof(vec3));}
    void *features;
    {MEMSAFE features = malloc(total_atoms * feature_size);}
    void *output;
    {MEMSAFE output = malloc(total_atoms * output_size);}
    ftype *weights;
    {MEMSAFE weights = (ftype *)malloc(total_atoms * sizeof(ftype));}
    pair<itype,itype> *edge_indices;
    {MEMSAFE edge_indices = (pair<itype,itype> *)malloc(total_edges * sizeof(pair<itype,itype>));}
    vec3 *edge_vecs;
    {MEMSAFE edge_vecs = (vec3 *)malloc(total_edges * sizeof(vec3));}

    int atom_tally = 0;
    int edge_tally = 0;
    for(Example *e : examples){
        int num_atoms = e->num_atoms;
        memcpy(positions + atom_tally, e->positions, num_atoms * sizeof(vec3));
        memcpy((char *)features + (feature_size * atom_tally), e->features, num_atoms * feature_size);
        memcpy((char *)output + (output_size * atom_tally), e->output, num_atoms * output_size);
        memcpy(weights + atom_tally, e->weights, num_atoms * sizeof(ftype));
        int num_edges = e->num_edges;
        memcpy(edge_vecs + edge_tally, e->edge_vecs, num_edges * sizeof(vec3));
        for(int edge = 0; edge < num_edges; edge++){
            pair<itype,itype> ei = e->edge_indices[edge];
            edge_indices[edge_tally + edge] = {ei.first + atom_tally, ei.second + atom_tally};
        }
        atom_tally += num_atoms;
        edge_tally += num_edges;
        {MEMSAFE delete e;}
    }

    {MEMSAFE return new Example(total_atoms, positions, features, output, weights,
            total_edges, edge_indices, edge_vecs, name, examples.size());}
}

void BatchGenerator::loopMakeBatches(int max) {
	for(int i=0; i != max; i++) {
        waitTillExComing();
        {
            unique_lock<mutex> bl(batch_count_mutex);
            num_batch++;
        }
        batch_queue.push(makeBatch());
    }
}

bool BatchGenerator::anyExComing(){
    unique_lock<mutex> lock(ex_count_mutex);
    while(!knows_ex_coming) know_cond.wait(lock);
    return num_ex > 0;
}

void BatchGenerator::waitTillExComing(){
    unique_lock<mutex> lock(ex_count_mutex);
    while(!knows_ex_coming || num_ex == 0) know_cond.wait(lock);
}

bool BatchGenerator::anyBatchComing(){
    {
        unique_lock<mutex> lk(batch_count_mutex);
        if(num_batch > 0) return true;
    }
    return anyExComing();
}

bool BatchGenerator::batchReady(){
    return batch_queue.size() > 0;
}

Example *BatchGenerator::getBatch(bool block) {
    if(!block){
        if(!batchReady()) return NULL;
    } else if(!anyBatchComing()) return NULL;
	Example *batch = batch_queue.pop(block);
    if(batch != NULL){
        unique_lock<mutex> lock(batch_count_mutex);
        num_batch--;
    }
    return batch;
}

void BatchGenerator::batchThreadRun(BatchGenerator* bg) {
	bg->loopMakeBatches();
}

void BatchGenerator::moleculeThreadRun(BatchGenerator* bg) {
    bg->loopProcessMolecules();
}

void BatchGenerator::notifyStarting(int bs){
    unique_lock<mutex> lock(ex_count_mutex);
    knows_ex_coming = false;
    finished_reading = false;
    if(bs > 0) batch_size = bs;
    know_cond.notify_all();
    //restart_cond.notify_all();
}

void BatchGenerator::notifyFinished(){
    unique_lock<mutex> lock(ex_count_mutex);
    knows_ex_coming = true;
    finished_reading = true;
    know_cond.notify_all();
    //restart_cond.notify_all();
}

int BatchGenerator::moleculeQueueSize(){
    return molecule_queue.size();
}
int BatchGenerator::exampleQueueSize(){
    return example_queue.size();
}
int BatchGenerator::batchQueueSize(){
    return batch_queue.size();
}

int BatchGenerator::numExample(){
    unique_lock<mutex> lock(ex_count_mutex);
    return num_ex;
}

int BatchGenerator::numBatch(){
    unique_lock<mutex> bl(batch_count_mutex);
    return num_batch;
}

void BatchGenerator::buildElementMap(vector<int> elements, vector<int> relevant_elements){
    for(int re : relevant_elements) this->relevant_elements.insert(re);
    num_elements = elements.size();
    //one_hots = (ftype *)calloc(num_elements * num_elements, sizeof(ftype));
    for(int i=0; i < num_elements; i++) element_map[elements[i]] = i;
}

BatchGenerator::BatchGenerator(int batch_size, float max_radius, int feature_size, int output_size,
        int num_threads, int molecule_cap, int example_cap, int batch_cap) : 
        max_radius(max_radius), feature_size(feature_size), output_size(output_size),
        molecule_queue(molecule_cap),
        example_queue(example_cap), 
        batch_queue(batch_cap)
{
	molecule_threads.reserve(num_threads);
    for(int i=0; i < num_threads; i++) molecule_threads.push_back(thread(moleculeThreadRun, this));
    batch_thread = thread(batchThreadRun, this);
}

BatchGenerator::BatchGenerator(int batch_size, float max_radius, vector<int> elements, vector<int> relevant_elements,
        int num_threads, int molecule_cap, int example_cap, int batch_cap) : 
        max_radius(max_radius), feature_size(elements.size() * sizeof(ftype)), output_size(sizeof(ftype)),
        molecule_queue(molecule_cap),
        example_queue(example_cap), 
        batch_queue(batch_cap)
{
    buildElementMap(elements, relevant_elements);
	molecule_threads.reserve(num_threads);
    for(int i=0; i < num_threads; i++) molecule_threads.push_back(thread(moleculeThreadRun, this));
    batch_thread = thread(batchThreadRun, this);
}

// Badly designed destructor. Doesn't properly clean things up. Just detaches threads and hopes for
// the best. Only intended to be run at program termination.
BatchGenerator::~BatchGenerator() {
	for(auto &t : molecule_threads) t.detach();
    batch_thread.detach();
}



// main function for testing without python:

int main() {
	{
		printf("Creating BatchGenerator...\n");
		BatchGenerator bg(100, 5, 32, 8, 4, 1000, 1000, 10);
        printf("Done.\n");
#ifdef _MSC_VER
	    system("pause");
#else
	    system("read");
#endif
		printf("Destroying BatchGenerator...\n");
	}
    printf("Done.\n");
#ifdef _MSC_VER
	system("pause");
#else
	system("read");
#endif
	
}