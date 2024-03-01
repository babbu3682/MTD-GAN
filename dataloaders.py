from create_datasets.Sinogram import *
from create_datasets.Mayo import *
from torch.utils.data import DataLoader

def get_train_dataloader(name, args):
    if name == 'amc':            
        train_dataset, train_collate_fn = Sinogram_Dataset_DCM(mode='train', type=args.dataset_type_train)
        valid_dataset, valid_collate_fn = Sinogram_Dataset_DCM(mode='valid', type=args.dataset_type_valid)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.train_num_workers, shuffle=True,  drop_last=True,  collate_fn=train_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=1,               num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=valid_collate_fn) 

    elif name == 'mayo':
        train_dataset, train_collate_fn = MAYO_Dataset_DCM(mode='train', type=args.dataset_type_train)
        valid_dataset, valid_collate_fn = MAYO_Dataset_DCM(mode='valid', type=args.dataset_type_valid)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.valid_num_workers, shuffle=True,  drop_last=True,  collate_fn=train_collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=1,               num_workers=args.valid_num_workers, shuffle=False, drop_last=False, collate_fn=valid_collate_fn) 

    else: 
        raise Exception('Error...! args.data_folder_dir')

    print("Train [Total] number = ", len(train_dataset))
    print("Valid [Total] number = ", len(valid_dataset))

    return train_loader, valid_loader


def get_test_dataloader(name, args):
    if name == 'amc':        
        test_dataset, test_collate_fn = TEST_Sinogram_Dataset_DCM(mode='test', type=args.dataset_type_test)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=test_collate_fn) 

    elif name == 'mayo':
        test_dataset, test_collate_fn = TEST_MAYO_Dataset_DCM(mode='test', type=args.dataset_type_test)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False, drop_last=False, collate_fn=test_collate_fn) 
    
    else: 
        raise Exception('Error...! args.data_set')

    print("Test [Total] number = ", len(test_dataset))

    return test_loader
