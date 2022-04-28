from datasets.Sinogram import *
    
def build_dataset(training_mode, args):

    if args.data_set == 'Sinogram_DCM':
        dataset, collate_fn = Sinogram_Dataset_DCM(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    elif args.data_set == 'Sinogram_NII': 
        dataset, collate_fn = Sinogram_Dataset_NII(mode=training_mode, patch_training=args.patch_training, multiple_GT=args.multiple_GT)
    elif args.data_set == 'Sinogram_Dataset_DCM_SACNN': 
        dataset, collate_fn = Sinogram_Dataset_DCM_SACNN(mode=training_mode, patch_training=args.patch_training)


    elif args.data_set == 'TEST_Sinogram_OLD': 
        dataset, collate_fn = TEST_Sinogram_Dataset_OLD(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)
    elif args.data_set == 'TEST_Sinogram_DCM': 
        dataset, collate_fn = TEST_Sinogram_Dataset_DCM()
    elif args.data_set == 'TEST_Sinogram_NII': 
        dataset, collate_fn = TEST_Sinogram_Dataset_NII()
    elif args.data_set == 'TEST_Sinogram_Dataset_DCM_SACNN': 
        dataset, collate_fn = TEST_Sinogram_Dataset_DCM_SACNN()
    else: 
        raise Exception('Error...! args.data_set')

    return dataset, collate_fn



