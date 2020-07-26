import numpy as np
import molecule_pipeline_ext

#as_tensor = None

class ExampleBatch():
    def __init__(self, pos, x, y, weights, edge_index, edge_attr, name):
        self.pos = pos
        self.x = x
        self.y = y
        self.weights = weights
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.name = name

class MoleculePipeline():
    def __init__(self, batch_size, max_radius, feature_size, output_size, num_threads = 2,
            molecule_cap = 10000, example_cap = 10000, batch_cap = 100):
        self.capsule = molecule_pipeline_ext.newBatchGenerator(batch_size, max_radius, feature_size,
            output_size, num_threads, molecule_cap, example_cap, batch_cap)
        self.batch_size = batch_size
        self.max_radius = max_radius
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_threads = num_threads
        self.molecule_cap = molecule_cap
        self.example_cap = example_cap
        self.batch_cap = batch_cap

    def notify_restarting(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        molecule_pipeline_ext.notifyStarting(self.capsule, batch_size)

    def notify_finished(self):
        molecule_pipeline_ext.notifyFinished(self.capsule)
        
    def put_molecule(self, m, block=True):
        return molecule_pipeline_ext.putMolecule(self.capsule, m.perturbed_geometries, m.features,
            m.perturbed_shieldings, m.weights, m.name, block)

    def batch_ready(self):
        return molecule_pipeline_ext.batchReady(self.capsule)

    def get_next_batch(self, block = True):
        r = molecule_pipeline_ext.getNextBatch(self.capsule, block)
        if r is None:
            return None
        pos, x, y, weights, edge_indexT, edge_attr, name = r
        (pos, x, y, weights, edge_index, edge_attr) = (as_tensor(pos), as_tensor(x), as_tensor(y),
                transpose(as_tensor(edge_indexT)), as_tensor(edge_attr))

        return ExampleBatch(pos, x, y, weights, edge_index, edge_attr, name)

    
if __name__ != '__main__':
    import torch
    from torch import as_tensor
    from torch import transpose

if __name__ == '__main__':
    as_tensor = lambda x: x
    from numpy import transpose
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
    bg = MoleculePipeline(batch_size = 100, max_radius = 5, feature_size = 40,
            output_size = 8, num_threads = 4, molecule_cap = 1000, example_cap = 1000, batch_cap = 10)
    print("Done.")
    pause()
    print("Exiting...")
