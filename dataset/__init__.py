

def get_dataloader(name,mode):
    if name=='hongkong':
        from .hongkong import Hongkong_dataset
        return Hongkong_dataset(mode)
    elif name=='amsterdam':
        from .amsterdam import Amsterdam_dataset
        return Amsterdam_dataset(mode)
    elif name=='global_hongkong':
        from .global_hongkong import Hongkong_dataset
        return Hongkong_dataset(mode)
    else:
        raise NotImplementedError('Dataset {} is not implemented'.format(name))