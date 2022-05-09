from importlib import import_module


class Data:
    def __init__(self, para, device_id):
        module = import_module('data.' + para.data_loader)
        self.dataloader_train = module.Dataloader(para, device_id, ds_type='train')
        self.dataloader_valid = module.Dataloader(para, device_id, ds_type='valid')
