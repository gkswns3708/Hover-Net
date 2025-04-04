normal_Config = {
    'dataset_path' : './dataset/training_data/consep/consep/train/540x540_164x164',
    'with_type': True, 
    'run_mode': 'train',
    'num_classes': 8,
    'seed': 1234,
    'shape_info' : {
        "train": {"input_shape": [270, 270], "mask_shape": [80, 80],}, 
        "valid": {"input_shape": [270, 270], "mask_sshape": [80, 80],},
        },
    'num_workers': 2,
    'batch_size': 16,
    'model_name': 'base_hovernet',
    'nr_type': 8, 
    'model_mode': 'original',
    'log_dir': './logs/',
    'train_dir_list': [
        "./dataset/training_data/consep/consep/train/540x540_164x164"
    ],
    'valid_dir_list': [
        "./dataset/training_data/consep/consep/valid/540x540_164x164"
    ],
    'pretrained': './pretrained/ImageNet-ResNet50-Preact_pytorch.tar',
    'dataset_name': 'consep',
    'num_epochs': 5,
}

uniform_Config = {  
}