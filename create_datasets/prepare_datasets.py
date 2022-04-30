from create_datasets.Sinogram import *
    
def build_dataset(training_mode, args):

    if args.data_folder_dir == '/workspace/sunggu/4.Dose_img2img/datasets/[sinogram]Brain_3mm_DCM':
        if args.multiple_GT:
            dataset, collate_fn = Sinogram_Dataset_DCM_Multiple_GT(mode=training_mode, patch_training=args.patch_training)
            
        elif args.windowing:
            dataset, collate_fn = Sinogram_Dataset_DCM_Windowing(mode=training_mode, patch_training=args.patch_training)

        else:
            dataset, collate_fn = Sinogram_Dataset_DCM(mode=training_mode, patch_training=args.patch_training)


    # elif args.data_set == 'Sinogram_Dataset_DCM_SACNN': 
        # dataset, collate_fn = Sinogram_Dataset_DCM_SACNN(mode=training_mode, patch_training=args.patch_training)

    else: 
        raise Exception('Error...! args.data_folder_dir')

    return dataset, collate_fn



def build_dataset_test(args):
    if args.data_folder_dir == 'TEST_Sinogram_OLD': 
        dataset, collate_fn = TEST_Sinogram_Dataset_OLD(mode=args.training_mode, range_minus1_plus1=args.range_minus1_plus1)
    elif args.data_folder_dir == 'TEST_Sinogram_DCM': 
        dataset, collate_fn = TEST_Sinogram_Dataset_DCM()

    # elif args.data_set == 'TEST_Sinogram_Dataset_DCM_SACNN': 
        # dataset, collate_fn = TEST_Sinogram_Dataset_DCM_SACNN()
    else: 
        raise Exception('Error...! args.data_set')

    return dataset, collate_fn