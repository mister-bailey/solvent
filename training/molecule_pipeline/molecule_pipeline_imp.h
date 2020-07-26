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

#define MEMSAFE lock_guard<mutex> lg(alloc_mutex); //if(this->end) return;

#define ftype double
#define itype long
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
    /*int feature_size;
    int output_size;*/
    PyArrayObject *positions;
    PyArrayObject *features;
    PyArrayObject *output;
    PyArrayObject *weights;
    string name;

    Molecule(int num_examples, int num_atoms, PyArrayObject *positions,
            PyArrayObject *features, PyArrayObject *output, PyArrayObject *weights, string name="");

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
    string name;

    Example(int num_atoms, vec3 *positions, void *features, void *output, ftype *weights,
            int num_edges, pair<itype,itype> *edge_indices, vec3 *edge_vecs, string name="");
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

	int batch_read = 0;
	int batch_write = 0;
	int batches_ahead = 2;

	vector<thread> molecule_threads;
    thread batch_thread;

	mutex alloc_mutex;
	bool go = true;
    mutex start_stop_mutex;
	bool end = false;

    mutex ex_count_mutex;
    bool knows_ex_coming = false;
    int num_ex = 0;
    condition_variable know_cond;
    mutex batch_count_mutex;
    int num_batch = 0;
    bool finished_reading = false;
    condition_variable restart_cond;


    bool anyExComing();

    ftype max_radius;

    void processMolecule(Molecule *molecule);
    void loopProcessMolecules(int max = -1);
    Example *makeBatch();
    void loopMakeBatches(int max = -1);

public:
    int feature_size;
    int output_size;

	//static BatchGenerator* makeBatchGenerator();
	static void batchThreadRun(BatchGenerator* bg);
    static void moleculeThreadRun(BatchGenerator* bg);

    void notifyStarting(int batch_size = -1);
    void notifyFinished();
    bool anyBatchComing();

	BatchGenerator(int batch_size, float max_radius, int feature_size, int output_size, int num_threads, int molecule_cap,
            int example_cap, int batch_cap, bool go = true);
	~BatchGenerator();

    bool putMolecule(Molecule *molecule, bool block=true);
    bool batchReady();
    Example *getBatch(bool block=true);

	void start();
	void stop();

	//int
};
