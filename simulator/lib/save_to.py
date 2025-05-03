import os
import datetime
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
            filename = '%d.csv'%ts
        else:
            filename = '%d.csv'%random_seed

        save_path = os.path.join(dirname, filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)

