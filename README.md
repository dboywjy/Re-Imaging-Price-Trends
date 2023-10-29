#### This is a replication for (Re-)Imag(in)ing Price Trends

You can download the paper:[(Re-)Imag(in)ing Price Trends](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)

The training log can be found in log.ipynb

baseline: model1.py + trainer1.py

extension:

- albation: model{2-11}.py + trainer{2-11}.py

- Grad Cam: Grad_CAM.ipynb, the output figures(down_sample_10.jpg, up_sample_10.jpg)

- regression: model_reg.py + trainer_reg.py (20d)
    model_reg.py + trainer_reg.py (5d)

each trainer has its own config in config.py

```
.
├── cnn_model
│   ├── trainer10_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer1_1_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer11_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer1_2_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer1_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer2_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer3_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer4_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer5_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer6_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer7_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer8_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer9_ret_20_classification_model_stat_checkpoint.pt
│   ├── trainer_reg2_ret_20_classification_model_stat_checkpoint.pt
│   └── trainer_reg_ret_20_classification_model_stat_checkpoint.pt
├── config.py
├── data
│   └── raw
│       └── raw.md
├── Grad_CAM.ipynb
├── log2.ipynb
├── log3.ipynb
├── log.ipynb
├── main.py
├── models
│   ├── __init___.py
│   ├── model10.py
│   ├── model11.py
│   ├── model1.py
│   ├── model2.py
│   ├── model3.py
│   ├── model4.py
│   ├── model5.py
│   ├── model6.py
│   ├── model7.py
│   ├── model8.py
│   ├── model9.py
│   ├── model_reg2.py
│   ├── model_reg.py
│   └── __pycache__
│       ├── model10.cpython-311.pyc
│       ├── model1_1.cpython-311.pyc
│       ├── model11.cpython-311.pyc
│       ├── model1_2.cpython-311.pyc
│       ├── model12.cpython-311.pyc
│       ├── model1.cpython-311.pyc
│       ├── model2.cpython-311.pyc
│       ├── model3.cpython-311.pyc
│       ├── model4.cpython-311.pyc
│       ├── model5.cpython-311.pyc
│       ├── model6.cpython-311.pyc
│       ├── model7.cpython-311.pyc
│       ├── model8.cpython-311.pyc
│       ├── model9.cpython-311.pyc
│       ├── model_reg2.cpython-311.pyc
│       └── model_reg.cpython-311.pyc
├── notebooks
│   ├── 1.0-wjy-initial-data-exploration.ipynb
│   ├── 2.0-wjy-initial-data-exploration.ipynb
│   ├── 3.0-wjy-initial-data-exploration.ipynb
│   └── 4.0-wjy-initial-data-exploration.ipynb
├── __pycache__
│   └── config.cpython-311.pyc
├── README.md
├── reports
│   └── figures
│       ├── down_sample_10.jpg
│       ├── train4_CNN_classification_loss.png
│       ├── train4_CNN_classification_train_val_loss.png
│       ├── train5_CNN_classification_loss.png
│       ├── train5_CNN_classification_train_val_loss.png
│       ├── trainer10_CNN_classification_loss.png
│       ├── trainer10_CNN_classification_train_val_loss.png
│       ├── trainer1_1_CNN_classification_loss.png
│       ├── trainer11_CNN_classification_loss.png
│       ├── trainer1_1_CNN_classification_train_val_loss.png
│       ├── trainer11_CNN_classification_train_val_loss.png
│       ├── trainer1_2_CNN_classification_loss.png
│       ├── trainer1_2_CNN_classification_train_val_loss.png
│       ├── trainer1_CNN_classification_loss.png
│       ├── trainer1_CNN_classification_train_val_loss.png
│       ├── trainer2_CNN_classification_loss.png
│       ├── trainer2_CNN_classification_train_val_loss.png
│       ├── trainer3_CNN_classification_loss.png
│       ├── trainer3_CNN_classification_train_val_loss.png
│       ├── trainer4_CNN_classification_loss.png
│       ├── trainer4_CNN_classification_train_val_loss.png
│       ├── trainer5_CNN_classification_loss.png
│       ├── trainer5_CNN_classification_train_val_loss.png
│       ├── trainer6_CNN_classification_loss.png
│       ├── trainer6_CNN_classification_train_val_loss.png
│       ├── trainer7_CNN_classification_loss.png
│       ├── trainer7_CNN_classification_train_val_loss.png
│       ├── trainer8_CNN_classification_loss.png
│       ├── trainer8_CNN_classification_train_val_loss.png
│       ├── trainer9_CNN_classification_loss.png
│       ├── trainer9_CNN_classification_train_val_loss.png
│       ├── trainer_reg2_CNN_classification_loss.png
│       ├── trainer_reg2_CNN_classification_train_val_loss.png
│       ├── trainer_reg_CNN_classification_loss.png
│       ├── trainer_reg_CNN_classification_train_val_loss.png
│       ├── traniner4_CNN_classification_loss.png
│       ├── traniner4_CNN_classification_train_val_loss.png
│       └── up_sample_10.jpg
├── trainers
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── trainer10.cpython-311.pyc
│   │   ├── trainer1_1.cpython-311.pyc
│   │   ├── trainer11.cpython-311.pyc
│   │   ├── trainer1_2.cpython-311.pyc
│   │   ├── trainer12.cpython-311.pyc
│   │   ├── trainer1.cpython-311.pyc
│   │   ├── trainer2.cpython-311.pyc
│   │   ├── trainer3.cpython-311.pyc
│   │   ├── trainer4.cpython-311.pyc
│   │   ├── trainer5.cpython-311.pyc
│   │   ├── trainer6.cpython-311.pyc
│   │   ├── trainer7.cpython-311.pyc
│   │   ├── trainer8.cpython-311.pyc
│   │   ├── trainer9.cpython-311.pyc
│   │   ├── trainer_reg2.cpython-311.pyc
│   │   └── trainer_reg.cpython-311.pyc
│   ├── trainer10.py
│   ├── trainer11.py
│   ├── trainer1.py
│   ├── trainer2.py
│   ├── trainer3.py
│   ├── trainer4.py
│   ├── trainer5.py
│   ├── trainer6.py
│   ├── trainer7.py
│   ├── trainer8.py
│   ├── trainer9.py
│   ├── trainer_reg2.py
│   └── trainer_reg.py
└── utils
    ├── data.py
    ├── __init__.py
    ├── __pycache__
    │   ├── data.cpython-311.pyc
    │   ├── __init__.cpython-311.pyc
    │   └── utils.cpython-311.pyc
    └── utils.py
```
