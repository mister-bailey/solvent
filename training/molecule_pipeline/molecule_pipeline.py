import numpy as np
import molecule_pipeline_ext
from torch import as_tensor, transpose
import sys

#as_tensor = None

class ExampleBatch():
    def __init__(self, pos, x, y, weights, edge_index, edge_attr, ID=-1, n_examples=1):
        self.pos = pos
        self.x = x
        self.y = y
        self.weights = weights
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.ID = ID
        self.n_examples = n_examples

    # this is in-place!
    def to(self, device):
        self.pos = self.pos.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.weights = self.weights.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self

    # this is in-place!
    def share_memory_(self):
        self.pos = self.pos.share_memory_()
        self.x = self.x.share_memory_()
        self.y = self.y.share_memory_()
        self.weights = self.weights.share_memory_()
        self.edge_index = self.edge_index.share_memory_()
        self.edge_attr = self.edge_attr.share_memory_()
        return self

class MoleculePipeline():
    def __init__(self, batch_size, max_radius, feature_size, output_size, num_threads = 2,
            molecule_cap = 10000, example_cap = 10000, batch_cap = 100, affine_dict = None):
        if isinstance(feature_size, int):
            self.capsule = molecule_pipeline_ext.newBatchGenerator(batch_size, max_radius, feature_size,
                    output_size, num_threads, molecule_cap, example_cap, batch_cap)
        else:
            self.capsule = molecule_pipeline_ext.newBatchGeneratorElements(batch_size, max_radius, feature_size,
                    output_size, num_threads, molecule_cap, example_cap, batch_cap, affine_dict)
            feature_size = len(feature_size)
            output_size = 1

        self.batch_size = batch_size
        self.max_radius = max_radius
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_threads = num_threads
        self.molecule_cap = molecule_cap
        self.example_cap = example_cap
        self.batch_cap = batch_cap

    def notify_starting(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        molecule_pipeline_ext.notifyStarting(self.capsule, batch_size)

    def notify_finished(self):
        molecule_pipeline_ext.notifyFinished(self.capsule)
        
    def put_molecule(self, m, block=True):
        return molecule_pipeline_ext.putMolecule(self.capsule, m.perturbed_geometries, m.features,
            m.perturbed_shieldings, m.weights, m.ID, block)

    # send data without first building a Molecule object.
    # Send atomic numbers, and 1-hots will be computed in C++
    def put_molecule_data(self, data, atomic_numbers, weights, ID, block=True):
        return molecule_pipeline_ext.putMoleculeData(self.capsule, data[...,:3], atomic_numbers, data[...,3], weights, ID, block)

    def batch_ready(self):
        return molecule_pipeline_ext.batchReady(self.capsule)

    def any_batch_coming(self):
        return molecule_pipeline_ext.anyBatchComing(self.capsule)

    def molecule_queue_size(self):
        return molecule_pipeline_ext.moleculeQueueSize(self.capsule)

    def example_queue_size(self):
        return molecule_pipeline_ext.exampleQueueSize(self.capsule)

    def batch_queue_size(self):
        return molecule_pipeline_ext.batchQueueSize(self.capsule)

    def num_example(self):
        return molecule_pipeline_ext.numExample(self.capsule)

    def num_batch(self):
        return molecule_pipeline_ext.numBatch(self.capsule)

    def get_next_batch(self, block = True):
        r = molecule_pipeline_ext.getNextBatch(self.capsule, block)
        if r is None:
            print("Batch is None!")
            return None
        pos, x, y, weights, edge_indexT, edge_attr, ID, n_examples = r
        (pos, x, y, weights, edge_index, edge_attr) = (as_tensor(pos), as_tensor(x), as_tensor(y),
                as_tensor(weights), transpose(as_tensor(edge_indexT),0,1), as_tensor(edge_attr))

        return ExampleBatch(pos, x, y, weights, edge_index, edge_attr, ID, n_examples)

    
#if __name__ != '__main__':
#    import torch
#    from torch import as_tensor
#    from torch import transpose

if __name__ == '__main__':
    #as_tensor = lambda x: x
    #from numpy import transpose
    import os
    import sys
    def pause():
        if sys.platform.startswith('win'):
            os.system('pause')
        else:
            os.system('read -s -n 1 -p "Press any key to continue..."')
    
    print("Imported molecule_pipeline")
    pause()
    print("Creating MoleculePipeline(batch_size = 100, max_radius = 5, feature_size = 40, output_size = 8,"
        "num_threads = 4, molecule_cap = 1000, example_cap = 1000, batch_cap = 10)...")
    mp = MoleculePipeline(batch_size = 100, max_radius = 5, feature_size = 40,
            output_size = 8, num_threads = 4, molecule_cap = 1000, example_cap = 1000, batch_cap = 10)
    print("Done.")
    pause()
    print("Exiting...")
