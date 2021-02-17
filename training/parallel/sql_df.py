import numpy as np
import pandas as pd
import sqlalchemy as sqa
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.expression import func
from sqlalchemy.schema import MetaData
from io import BytesIO

# mysql://ekwan16:h9#Li48Z#hY$b@J8@SG-nmrdatabase-2962-master.servers.mongodirector.com/pbe0

# status code meanings:
# 0 - "not started" meaning only one jiggle is present
# 1 - "complete" meaning 0th entry in data column is stationary, 1st entry is jiggle
# 2 - "pending" meaning the stationary is currently being computed
# 3 - "error" meaning something went wrong and this row is considered dead

def connect(func):
    def wrapped(self, *args, **kwargs):
        connect = True if self.connection is None else False
        if connect: self.__enter__()
        r = func(self, *args, **kwargs)
        if connect: self.__exit__(None, None, None)
        return r
    return wrapped
    
# converts a byte representation of a numpy array to an actual numpy array
def unpack_bytes(arr_bytes):
    load_bytes = BytesIO(arr_bytes)
    loaded_np = np.load(load_bytes, allow_pickle=True)
    return loaded_np

class Database:
    
    def __init__(self, host, user, passwd=None, db="pbe0", status=0):
        self.host = host
        self.user = user
        self.passwd = "" if passwd is None else ":" + passwd
        self.db = db
        self.dialect = "mysql+pymysql"
        self.status = status        
        self.metadata = MetaData()
        
        self.connection = None
        self.engine = None
        
        self.__enter__()
        self.status_table = sqa.Table('status_new', self.metadata, autoload=True, autoload_with=self.engine)
        self.data_table = sqa.Table('data_new', self.metadata, autoload=True, autoload_with=self.engine)
        self.__exit__(None, None, None)
        
        
        
        
    def __enter__(self):
        if self.connection is not None: return
        self.engine = sqa.create_engine(f"{self.dialect}://{self.user}{self.passwd}@{self.host}/{self.db}",
                connect_args={'ssl':{'ssl': {}}}, poolclass=NullPool)
        self.connection = self.engine.connect()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection is None: return
        self.connection.close()
        self.connection = None
        self.engine.dispose()
        self.engine = None
    
    @connect   
    def fetch_ids(self, number=None, requested_status=None, increment=10000, verbose=True):
        if requested_status is None:
            requested_status = self.status
        query = sqa.select([self.status_table.columns.id]).where(self.status_table.columns.status == requested_status)
        if number:
            query = query.limit(number)
            num = f"{number:,d}"
        else:
            num = "ALL"
        if verbose: print(f"Fetching {num} IDs with status {requested_status}...", flush=True)
        result = self.connection.execution_options(stream_results=True).execute(query)
        
        rows = []
        batch = True
        while batch:
            batch = result.fetchmany(increment)
            rows += batch
            if verbose: print(f" {len(rows):,d} / {num} \r", flush=True, end='')
            
        return np.array([int(r[0]) for r in rows]) 
        
    @connect
    def read_rows(self, ids, columns=['id', 'atomic_numbers', 'geometries_and_shieldings', 'compound_type', 'weights'], randomize=False):
        query = sqa.select((getattr(self.data_table.columns, c) for c in columns))
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        query = query.where(self.data_table.columns.id.in_(ids))
        if randomize:
            query = query.order_by(func.rand())
        
        query_df = pd.read_sql_query(query, self.engine)
        self.__exit__(None, None, None)
        
        # convert back to the array types
        query_df.atomic_numbers = query_df.atomic_numbers.apply(unpack_bytes)
        query_df.geometries_and_shieldings = query_df.geometries_and_shieldings.apply(unpack_bytes)
        #query_df.symmetric_atoms = query_df.symmetric_atoms.apply(ast.literal_eval)
        query_df.weights = query_df.weights.apply(unpack_bytes)
        
        return query_df
        

if __name__ == '__main__':
    db = Database('SG-nmrdatabase-2962-master.servers.mongodirector.com', 'ekwan16', 'h9#Li48Z#hY$b@J8', 'pbe0')
    
    ids = db.fetch_ids(10)
    print('\n', list(ids))
    
    print("\nRetrieving IDs as rows...\n")
    data = db.read_rows(ids)
    
    print(data)
    

    
        
