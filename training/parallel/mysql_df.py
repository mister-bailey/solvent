import numpy as np
import os, time, re, ast, pymysql, yaml
from getpass import getpass

from collections import Counter

def inflate(s, dtype=np.float64):
    s = re.sub('\s+', ',', s)
    s = re.sub('\[,', '[', s)
    return np.array(ast.literal_eval(s), dtype=dtype)

def inflate_x(s):
#    if s is None:
#        return None
#    if not re.search(",", s):
#        s = re.sub('\s+', ',', s)
#        s = re.sub('\[,', '[', s)
    s = re.sub("array\(", "", s)
    s = re.sub("\)", "", s)
    return np.array(ast.literal_eval(s))


class MysqlDB():
    """
    Represents an ``.db`` datafile.
    Object attributes:
        connect_params:
            YAML file containing params needed to connect to database (not stored in memory to permit facile changes).
        smiles_path:
            path to text file containing a list of smiles strings (one per line)
        path:
            path to logfile
    Rows of db:
        data:
        energy:
        e_rel:
        gdb_id:
            smiles of molecule
        status:
            0 - not yet begun
            1 - completed
            2 - pending
            3 - error
    """

    def __init__(self, connect_params, smiles_path=None, path=None, create_table=False):
        if isinstance(connect_params, str):
            assert os.path.exists(connect_params)
            with open(connect_params, "r+") as f:
                self.connect_params = yaml.safe_load(f)
        else:
            self.connect_params = connect_params
        if 'passwd' not in self.connect_params:
            print(f"Using database {self.connect_params['db']}: {self.connect_params['user']}@{self.connect_params['host']}")
            self.connect_params['passwd'] = getpass(prompt="Please enter password: ")
        if 'ssl' not in self.connect_params:
            self.connect_params['ssl'] = {'ssl':{}}

        #assert isinstance(smiles_path, str), "path must be a string"
        self.smiles_path = smiles_path
        #assert os.path.exists(smiles_path)

        self.path = path

        # build table
        if create_table:
            con = pymysql.connect(**self.connect_params)
            with con.cursor() as cursor:
                cursor.execute("create table if not exists data (id integer primary key, data longtext, energy float, e_rel float, gdb_id tinytext, status int)")
            con.commit()
            con.close()

#        con = pymysql.connect(**self.connect_params)

    def read(self, row_indices):
        """
        Returns data, energy, e_rel, gdb_id, and status.
        """

        values = [None] * 7

        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute(f"select data, energy, e_rel, gdb_id, status, symmetric_atoms, weights from data where id in ({','.join(['%s']*len(row_indices))})", row_indices)
            rows = cursor.fetchall()

            for i in range(len(values)):
                if i == 0 or i == 6:
                    values[i] = [inflate(r[i]) for r in rows]
                else:
                    values[i] = [r[i] for r in rows]
        con.close()
        return values

    def read_rows(self, row_indices, randomize=True, get_tensors=False, check_status=False, debug_tensors=False):
        """
        Returns list of tuples (id, data, weights, smiles).
        """
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select id, data, weights, gdb_id{', tensors' if get_tensors else ''} from data where "
            if check_status:
                command += "status = 1 and weights is not null and "
            command += "id in (" + ",".join(map(str, row_indices)) + ")"
            if randomize:
                command += " order by rand()"
            #f"select id, data, weights, gdb_id from data where status = 1 and weights is not null and id in ({','.join(['%s']*len(row_indices))})", row_indices
            result = cursor.execute(command)
            rows = cursor.fetchall()
        con.close()
        
        if get_tensors:
            if debug_tensors:
                for i,r in enumerate(rows[:10]):
                    data = inflate(r[1])
                    weights = inflate(r[2])
                    tensors = inflate_x(r[4])
                    print(f"data[{r[0]}]: {list(data.shape)},  weights[.]: {list(weights.shape)},  tensors[.]: {list(tensors.shape)}")
            #value = []
            #for r in rows:
            #    r1, r4 = inflate(r[1]), inflate_x(r[4])
            #    if r1.shape[0] == r4.shape[0]:
            #        value.append((r[0], np.concatenate((r1, r4.reshape((-1,9))), axis=1), inflate(r[2]), r[3]))
            #return value
            return [(r[0], np.concatenate((inflate(r[1]), inflate_x(r[4]).reshape((-1,9))), axis=1), inflate(r[2]), r[3]) for r in rows] # if ((not check_status) or r[4] == 1)] 
        else:
            return [(r[0], inflate(r[1]), inflate(r[2]), r[3]) for r in rows] # if ((not check_status) or r[4] == 1)]
        

    def read_range(self, start_row, stop_row, check_status=True, randomize=True, limit=None):
        """
        Returns list of tuples (id, data, weights, smiles).
        """

        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select id, data, weights, gdb_id from data where id >= {start_row} and id < {stop_row}"
            if check_status:
                command += " and status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            rows = cursor.fetchall()

        con.close()
        return [(r[0], inflate(r[1]), inflate(r[2]), r[3]) for r in rows] #if r[4] == 1]


    def write(self, row_idxs, data, energy, e_rel, gdb_id, status, check=True):
        """
        Writes new rows.
        """
        data = [str(d) for d in data]

        smileses = list()
        for smiles in gdb_id:
            if not isinstance(smiles, str):
                if smiles is None:
                    raise ValueError(f"{smiles} is null, not a smiles string")
                smiles = smiles.decode("UTF-8")
            smileses.append(smiles)

        row_idxs = [int(i) for i in row_idxs]
        energy = [float(e) for e in energy]
        e_rel = [float(e) for e in e_rel]
        status = [int(s) for s in status]

        wrote_any = False
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            for vals in list(zip(row_idxs, data, energy, e_rel, smileses, status)):
                if check:
                    smiles = vals[4]

                    result = cursor.execute("select count(*) from data where gdb_id = %s and status = 1", [smiles])
                    if cursor.fetchall()[0][0]:
                        continue
                    else:
                        cursor.execute("replace into data (id, data, energy, e_rel, gdb_id, status) values (%s, %s, %s, %s, %s, %s)", vals)
                        wrote_any = True
                else:
                    cursor.execute("replace into data (id, data, energy, e_rel, gdb_id, status) values (%s, %s, %s, %s, %s, %s)", vals)
                    wrote_any = True
        if wrote_any:
            con.commit()
        con.close()
        return wrote_any

    def status_report(self):
        """
        Returns aggregate status of all rows.
        """
        counter = dict()
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute("select status, count(status) from data group by status")
            for row in result.fetchall():
                counter[row[0]] = row[1]
        con.close()
        return counter

    def check_status(self, id):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute(f"select status from data where id = {id}")
            row = cursor.fetchall()

        con.close()
        return row[0][0]


    def set_status(self, row_idxs, status_id, check=True):
        """
        """
        wrote_any = False
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            for i in row_idxs:
                print(f"set status for {i}")
                if check:
                    cursor.execute("select count(*) from data where id = %s and status = 1", (i))
                    result = cursor.fetchall()[0][0]
                    print(result)
                    if result:
                        continue
                    else:
                        print("executing")
                        cursor.execute("replace into data (id, status) values (%s, %s)", (i, status_id))
                        wrote_any = True
                else:
                    cursor.execute("replace into data (id, status) values (%s, %s)", (i, status_id))
                    wrote_any = True
        if wrote_any:
            con.commit()
        con.close()
        return wrote_any

    def get_smiles_from_index(self, row_index):
        smiles = None
        with open(self.smiles_path) as f:
            for idx, row in enumerate(f):
                if idx == row_index:
                    smiles = row.strip()
                    break
        return smiles

    def next_rows(self, num, which="new"):
        """
        Returns index of the first ``num`` rows not yet begun.
        Returns -1 if all rows are complete.
        """
        idxs = list()
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            if which == "new":
                result = cursor.execute("select id from data")
                if result:
                    idxs = [x[0] for x in cursor.fetchall()]
            elif which == "failed":
                result = cursor.execute("select id from data where status = 1")
                if result:
                    idxs = [x[0] for x in cursor.fetchall()]
            else:
                raise ValueError("which which?")
        con.close()
        next_idxs = list()
        current_idx = 0
        while len(next_idxs) < num:
            if current_idx not in idxs:
                next_idxs.append(current_idx)
            current_idx += 1
        return next_idxs

    def expunge(self, block_size=1e4):
        """
        Collapses all status values greater than 1 to 0.
        """
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute("delete from data where status != 1")
        con.commit()
        con.close()

    def get_finished_idxs(self, limit=None, ordered=False):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = "select data.id from data inner join status_log on data.id = status_log.id where status_log.status = 1 and data.weights is not null"
            if ordered:
                command += " order by id asc"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            idxs = [x[0] for x in cursor.fetchall()]
        con.close()
        return idxs


    def get_unfinished_idxs(self, start_row, stop_row):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute(f"select id from data where id >= {start_row} and id < {stop_row} and (status != 1 or weights is null) order by id asc")
            idxs = [x[0] for x in cursor.fetchall()]
        con.close()
        return idxs

    def get_distinct_columns(self, column, limit=None, check_status=False):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select distinct {column} from data"
            if check_status:
                command += "where status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            cols = [row[0] for row in cursor.fetchall()]
        con.close()
        return cols

    def get_rows_from_column(self, column, value, limit=None, check_status=False):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select id, data, weights, gdb_id from data where {column} = \"{value}\""
            if check_status:
                command += " and status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            rows = cursor.fetchall()

        con.close()
        return [(r[0], inflate(r[1]), inflate(r[2]), r[3]) for r in rows]
    
    def get_columns_from_column(self, get_column, where_column, value, limit=None, check_status=False):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select {get_column} from data where {where_column} = \"{value}\""
            if check_status:
                command += " and status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            cols = [row[0] for row in cursor.fetchall()]

        con.close()
        return cols

    def get_column_from_range(self, column, start, stop, limit=None, check_status=True):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select {column} from data where id >= {start} and id < {stop}"
            if check_status:
                command += " and status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            cols = [row[0] for row in cursor.fetchall()]

        con.close()
        return cols

    def get_columns_with_cond(self, column, cond, limit=None, check_status=True):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select {column} from data where {cond}"
            if check_status:
                command += " and status = 1 and weights is not null"
            if isinstance(limit, int):
                command += f" limit {limit}"
            result = cursor.execute(command)
            cols = [row[0] for row in cursor.fetchall()]

        con.close()
        return cols




    def get_raw_rows(self, limit):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            command = f"select * from data limit {limit}"
            result = cursor.execute(command)
            cols = list(cursor.fetchall())

        con.close()
        return cols

    def get_column_names(self):
        con = pymysql.connect(**self.connect_params)
        with con.cursor() as cursor:
            result = cursor.execute("show columns from data")
            cols = list(cursor.fetchall())

        con.close()
        return cols





if __name__ == '__main__':
    # Test
    db = MysqlDB("local_mysql.yaml", None, None)
    print(f"Connected to {db.connect_params['db']}: {db.connect_params['user']}@{db.connect_params['host']}")
    #rs = db.read_range(0,20,check_status=False)
    #for r in rs: print(r) 
    print(db.get_column_names())
    rs = db.get_raw_rows(1)
    print(rs[0])