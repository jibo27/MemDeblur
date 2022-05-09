import os
import shutil
from datetime import datetime
from os.path import dirname, join, basename

import torch


class Logger():
    def __init__(self, para, config_path=None):
        self.para = para
        self.config_path = config_path
        now = datetime.now() if 'time' not in vars(para) else para.time
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        file_path = join(para.save_dir, para.exp_name, now, 'log.txt')
        self.save_dir = dirname(file_path)
        if para.test_only:
            print('test_only')
            os.makedirs(para.test_save_dir, exist_ok=True)
            self.logger = open(join(para.test_save_dir, 'log.txt'), 'a+')
        else:
            self.check_dir(file_path)
            if self.config_path is not None:
                shutil.copy2(self.config_path, self.save_dir)
            self.logger = open(file_path, 'a+')
        self.register_dict = {}

    def record_para(self):
        self('recording parameters ...')
        for key, value in vars(self.para).items():
            self('{}: {}'.format(key, value), timestamp=False)

    def check_dir(self, file_path):
        dir = dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, verbose=True, prefix='', timestamp=True):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        info = prefix + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()

    # register values for each epoch, such as loss, PSNR etc.
    def register(self, name, epoch, value):
        if name in self.register_dict:
            self.register_dict[name][epoch] = value
            if value > self.register_dict[name]['max']:
                self.register_dict[name]['max'] = value
            if value < self.register_dict[name]['min']:
                self.register_dict[name]['min'] = value
        else:
            self.register_dict[name] = {}
            self.register_dict[name][epoch] = value
            self.register_dict[name]['max'] = value
            self.register_dict[name]['min'] = value

    def report(self, items, state, epoch):
        # items - [['MSE', 'min'], ['PSNR', 'max'] ... ]
        msg = '[{}] '.format(state.lower())
        state = '_' + state.lower()
        for i in range(len(items)):
            item, best = items[i]
            msg += '{} : {:.4f} (best {:.4f})'.format(
                item,
                self.register_dict[item + state][epoch],
                self.register_dict[item + state][best]
            )
            if i < len(items) - 1:
                msg += ', '
        self(msg, timestamp=False)

    def is_best(self, epoch):
        item = self.register_dict[self.para.loss + '_valid']
        return item[epoch] == item['min']

    def save(self, state, filename='checkpoint.pth.tar'):
        path = join(self.save_dir, filename)
        torch.save(state, path)
        if self.is_best(state['epoch']):
            copy_path = join(self.save_dir, 'model_best.pth.tar')
            shutil.copy(path, copy_path)

