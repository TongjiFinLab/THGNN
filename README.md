# Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction (THGNN)
## 1. Prepare you training data
The input to your model is a pkl file that includes the stock symbol `code`, the time `dt`, and the volume and price features. Then, you need to use `generate_relation.py` to generate daily stock relationships and `generate_data.py` to generate the final input data for the model. You can adjust the features used in building the stock relationship and generating the final input by changing `feature_cols`. The `relation` directory stores the relations between stocks. The `daily_stock` directory contains stocks that are trained each day. The `data_train_predict` directory stores the final inputs fed to the model each day. The `prediction` directory stores the prediction result of the validation set. The `model_saved` directory stores the trained model.

## 2. Train you model
* Before training, make sure to change the parameters in class `Args` and function `main`.

 ``` config
 adj_threshold = 0.1         # the threshold of the relations between stocks
 max_epochs = 60             # the number of training epochs
 epochs_eval = 10            # the number of training epochs per evaluation or test interval
 epochs_save_by = 60         # the number of training epochs before a model is saved
 lr = 0.0002                 # learning rate of the model
 gamma = 0.3                 # gamma
 hidden_dim = 128            # hidden_dim
 num_heads = 8               # num_heads
 out_features = 32           # out_features
 model_name = "StockHeteGAT" # The main model name in model.thgnn.py
 dropout = 0.1               # dropout
 batch_size = 1              # batch_size
 loss_fcn = mse_loss         # loss function
 epochs_save_by = 60         # the number of training epochs of the saved model
 data_start = 20             # index of training start date
 data_middle = 39            # index of evaluation or test start date/ index of training end date
 data_end = data_middle+4    # index of evaluation or test end date
 pre_data = '2021-12-29'     # save the last date of the training
 ```

* Install required packages

  ``` shell
  pip install -r requirements.txt  for specific versions
  ```

* Training 

  ``` shell
  sh train.sh
  ```
## 3. Citing

* If you find **THGNN** is useful for your research, please consider citing the following papers:

  ``` latex
  @inproceedings{xiang2022thgnn,
      title={Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
      author={Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
      booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      pages = {3584–3593},
	  numpages = {10},
      year={2022},
      doi = {10.1145/3511808.3557089}
    }
  ```