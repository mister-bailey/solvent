import time

# utility class for timing operations
class Stopwatch():
    def __init__(self, running=False):
        self.elapsed = 0
        self.start_time = None
        self.running = False
        if running:
            self.start()
            
    def start(self):
        assert not self.running, "can't start a stopwatch that's already started"
        self.start_time = time.time()
        self.running = True
        
    def stop(self):
        assert self.running, "can't stop a stopwatch that isn't running"
        now = time.time()
        self.elapsed += now - self.start_time
        self.start_time = None
        self.running = False
        
    def get_elapsed(self):
        if self.running:
            now = time.time()
            elapsed = self.elapsed + now - self.start_time
            return elapsed
        else:
            return self.elapsed
    
    def reset(self):
        self.elapsed = 0
        self.running = False
        self.start_time = None

