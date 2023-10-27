IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}  

train_val_years = list(range(1993, 1999+1))
test_years = list(range(1999+1, 2019+1))

trainer1 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer1_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer2 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer2_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer3 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer3_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer4 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer4_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer5 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer5_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer6 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer6_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer7 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer7_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer8 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer8_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer9 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer9_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer10 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer10_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer11 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.00001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.000001
    ,'early_stopping_path':'./cnn_model/trainer11_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d_label'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer_reg = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.0000001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.00000001
    ,'early_stopping_path':'./cnn_model/trainer_reg_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_20d'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

trainer_reg2 = {
    'batch_size':128
    ,'init_lr':0.00001
    ,'sheduler_factor':0.8
    ,'sheduler_patience':5
    ,'sheduler_threshold':0.0000001
    ,'early_stopping_patience':2
    ,'early_stopping_delta':0.00000001
    ,'early_stopping_path':'./cnn_model/trainer_reg2_ret_20_classification_model_stat_checkpoint.pt'
    ,'epoch':100
    ,'target':'Retx_5d'
    ,'path':'/home/jwangiy/Reimage/img_data/monthly_20d'
}

