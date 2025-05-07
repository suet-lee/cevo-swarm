import os
import datetime
import json
import pandas as pd

class GenDir:

    BASE = 'data'

    def __init__(self):
        self.ts = int(datetime.datetime.now().timestamp())

    def mkdir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def gen_save_dirname(self, ex_id, sub_dir=None, makedir=True):
        dirname = '%s'%(ex_id)
        target = os.path.join(self.BASE, dirname)
        if sub_dir is not None:
            target = os.path.join(target, sub_dir)
        if makedir:
            self.mkdir(target)
        return target

class SaveTo(GenDir):

    def export_data(self, ex_id, data, filename, transpose=False):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if transpose:
            data = data.transpose()

        dirname = self.gen_save_dirname(ex_id, str(self.ts))
        save_path = os.path.join(dirname, '%s.csv'%filename)
        try:
            data.to_csv(save_path, index=False)
        except Exception as e:
            print(e)
        finally:
            return dirname

    def export_metadata(self,dirname,ex_cfg):
        fn = os.path.join(dirname,'metadata.txt')
        with open(fn,"w") as f:
            f.write(json.dumps(ex_cfg))
