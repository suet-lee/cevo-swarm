import os
import datetime
import json
import pandas as pd

class GenDir:

    BASE = 'data'

    def mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def gen_save_dirname(self, ex_id, makedir=True):
        dirname = '%s'%(ex_id)
        target = os.path.join(self.BASE, dirname)
        if makedir:
            self.mkdir(target)
        return target

class SaveTo(GenDir):

    def export_data(self, ex_id, data, random_seed=None):
        ts = datetime.datetime.now().timestamp()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        dirname = self.gen_save_dirname(ex_id)
        if random_seed is None:
            filename = ts
        else:
            filename = random_seed

        save_path = os.path.join(dirname, '%d.csv'%filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)
        finally:
            return dirname, filename

    def export_data2(self, ex_id, data, data_name, random_seed=None):
        ts = datetime.datetime.now().timestamp()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        dirname = self.gen_save_dirname(ex_id)
        if random_seed is None:
            filename = data_name+'_%d.csv'%ts
        else:
            filename = data_name+'_%d.csv'%random_seed

        save_path = os.path.join(dirname, filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)

    def export_metadata(self,dirname,filename,ex_cfg):
        fn = os.path.join(dirname,'%d.txt'%filename)
        with open(fn,"w") as f:
            f.write(json.dumps(ex_cfg))
