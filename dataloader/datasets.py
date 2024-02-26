from dataloader.dexycb_dataloader import DexYCBDataset
from dataloader.ho3d_dataloader import HO3DDataset

def get_dataset(dataset_name, model_name):
    if dataset_name == "dexycb":
        data_train = DexYCBDataset(setup = 's0', split ='training', model=model_name)
        data_val = DexYCBDataset(setup = 's0', split ='val', model=model_name)
        data_test = DexYCBDataset(setup = 's0', split ='test', model=model_name)
    elif dataset_name == "ho3d":
        data_train = HO3DDataset(setup = 's0', split ='training', model=model_name)
        data_val = HO3DDataset(setup = 's0', split ='val', model=model_name)
        data_test = HO3DDataset(setup = 's0', split ='test', model=model_name)       
    return data_train, data_val, data_test