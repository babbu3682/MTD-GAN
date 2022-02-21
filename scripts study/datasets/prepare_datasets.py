from datasets.Sinogram import *
    
def build_dataset(training_mode, args):

    if args.data_set == 'Sinogram_DCM':
        dataset, collate_fn = Sinogram_Dataset_DCM(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    elif args.data_set == 'Sinogram_NPY': 
        dataset, collate_fn = Sinogram_Dataset_NPY(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    elif args.data_set == 'Sinogram_NII': 
        dataset, collate_fn = Sinogram_Dataset_NII(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    elif args.data_set == 'Mayo_DCM':
        dataset, collate_fn = Mayo_Dataset_DCM(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    
   
    elif args.data_set == 'TEST_Sinogram_OLD': 
        dataset, collate_fn = TEST_Sinogram_Dataset_OLD(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)
    elif args.data_set == 'TEST_Sinogram_DCM': 
        dataset, collate_fn = TEST_Sinogram_Dataset_DCM(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)
    elif args.data_set == 'TEST_Sinogram_NII': 
        dataset, collate_fn = TEST_Sinogram_Dataset_NII(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)
    elif args.data_set == 'TEST_Mayo_DCM': 
        dataset, collate_fn = TEST_Mayo_Dataset_DCM(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)

    else: 
        raise Exception('Error...! args.data_set')

    return dataset, collate_fn



