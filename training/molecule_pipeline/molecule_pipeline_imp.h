/******************************************************************
Copyright Michael Bailey 2020
******************************************************************/
#pragma once
#define molecule_pipeline_imp

#include <vector>
#include <tuple>
#include <utility>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <climits>
#include <string>
//#include <Python.h>
#include <arrayobject.h>
#include <stdint.h>
#include <map>
#include <set>
#include <stdio>

#define MEMSAFE lock_guard<mutex> lg(alloc_mutex); //if(this->end) return;
#define ext_malloc(x) PyArray_malloc(x)
#define ext_free(x) PyArray_free(X)

// Puts in ivisible file/line outpus
#define LINETRACK printf(stderr, "Execution stopping on Line %d in %s\r", __LINE__, __FILE__)

#define ftype double //NPY_FLOAT64
#define itype int64_t
#define vec3 tuple<ftype,ftype,ftype>
#define data(aop) PyArray_DATA(aop)
#define data3(aop) (vec3 *)PyArray_DATA(aop)
#define nbytes(aop) PyArray_NBYTES(aop)

using namespace std;

inline vec3 sum(vec3 x, vec3 y){
    return vec3(get<0>(x) + get<0>(y), get<1>(x) + get<1>(y), get<2>(x) + get<2>(y));
}

inline vec3 diff(vec3 x, vec3 y){
    return vec3(get<0>(x) - get<0>(y), get<1>(x) - get<1>(y), get<2>(x) - get<2>(y));
}

inline ftype norm2(vec3 v){
    return get<0>(v) * get<0>(v) + get<1>(v) * get<1>(v) + get<2>(v) * get<2>(v);
}

class Molecule {
public:
    int num_examples;
    int num_atoms;
    PyArrayObject *positions;
    PyArrayObject *features;
    PyArrayObject *output;
    PyArrayObject *weights;
    int ID;

    Molecule(int num_examples, int num_atoms, PyArrayObject *positions,
            PyArrayObject *features, PyArrayObject *output, PyArrayObject *weights, int ID=-1);

    Molecule(int num_examples, int num_atoms, PyArrayObject *positions,
            const itype *elements, PyArrayObject *output, PyArrayObject *weights, int ID=-1);

    ~Molecule();
};

class Example {
public:
    int num_atoms;
    /*int feature_size;
    int output_size;*/
    vec3 *positions;
    void *features;
    void *output;
    ftype *weights;
    int num_edges;
    pair<itype,itype> *edge_indices;
    vec3 *edge_vecs;
    int ID;
    int n_examples = 1;

    Example(int num_atoms, vec3 *positions, void *features, void *output, ftype *weights,
            int num_edges, pair<itype,itype> *edge_indices, vec3 *edge_vecs, int ID=-1, int n_examples=1);
    ~Example();
    void releaseBuffers();
};

template <typename T>
class SynchronisedQueue {
private:
  queue<T> s_queue;     
  mutex q_mutex;            // The mutex to synchronise on
  condition_variable empty_cond;    // waits while queue is empty
  condition_variable full_cond;     // waits while queue is full
  int capacity;
  T is_empty; // returns from non-blocking calls to an empty queue
public:
    //SynchronisedQueue(int capacity = INT_MAX) : capacity(capacity) { }
    SynchronisedQueue(int capacity = INT_MAX, T is_empty = (T)NULL) : capacity(capacity), is_empty(is_empty) { }
    virtual ~SynchronisedQueue() throw () { }
    // Add data to the queue and notify poppers
    bool push(const T& x, bool block=true){
        unique_lock<mutex> lock(q_mutex);
        if(!block && s_queue.size() >= capacity) return false;
        while(s_queue.size() >= capacity) full_cond.wait(lock);
        s_queue.push(x);
        empty_cond.notify_one();
        return  true;
    }
    // Get data from the queue. Wait for data if not available
    T pop(bool block=true)
    {
      unique_lock<mutex> lock(q_mutex);
      if(!block && s_queue.size()==0) return is_empty;
      // When there is no data, wait till someone fills it.
      while (s_queue.size()==0) empty_cond.wait(lock);
      // Retrieve the data from the queue
      T result=s_queue.front(); s_queue.pop();
      full_cond.notify_one(); // let pushers know the queue has room
      return result;
    } // Lock is automatically released here

    int size(){
        unique_lock<mutex> lock(q_mutex);
        return s_queue.size();
    }
  };


class BatchGenerator {

	int batch_size; // In examples

    SynchronisedQueue<Molecule *> molecule_queue;
    SynchronisedQueue<Example *> example_queue;
    SynchronisedQueue<Example *> batch_queue;

	vector<thread> molecule_threads;
    thread batch_thread;

	mutex alloc_mutex;

    mutex ex_count_mutex;
    bool knows_ex_coming = false;
    int num_ex = 0;
    condition_variable know_cond;
    mutex batch_count_mutex;
    int num_batch = 0;
    bool finished_reading = false;

    void buildElementMap(vector<int> elements, vector<int> relevant_elements);

    bool anyExComing();
    void waitTillExComing();

    ftype max_radius;

    void processMolecule(Molecule *molecule);
    void loopProcessMolecules(int max = -1);
    Example *makeBatch();
    void loopMakeBatches(int max = -1);

public:
    int feature_size;
    int output_size;

    static int num_elements;
    static map<int, int> element_map;
    static set<int> relevant_elements;

	//static BatchGenerator* makeBatchGenerator();
	static void batchThreadRun(BatchGenerator* bg);
    static void moleculeThreadRun(BatchGenerator* bg);

    void notifyStarting(int batch_size = -1);
    void notifyFinished();
    bool anyBatchComing();

	BatchGenerator(int batch_size, float max_radius, int feature_size, int output_size, int num_threads, int molecule_cap,
            int example_cap, int batch_cap);
	BatchGenerator(int batch_size, float max_radius, vector<int> elements, vector<int> relevant_elements, int num_threads, int molecule_cap,
            int example_cap, int batch_cap);
	~BatchGenerator();

    bool putMolecule(Molecule *molecule, bool block=true);
    bool batchReady();
    Example *getBatch(bool block=true);

	//void start();
	//void stop();

    int moleculeQueueSize();
    int exampleQueueSize();
    int batchQueueSize();
    int numExample();
    int numBatch();

    SynchronisedQueue<Molecule *> deletion_queue;

	//int
};

