# Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction (THGNN)
## 1. Prepare you training data
The input to your model is a pkl file that includes the stock symbol `code`, the time `dt`, and the volume and price features. Then, you need to use `generate_relation.py` to generate daily stock relationships and `generate_data.py` to generate the final input data for the model. You can adjust the features used in building the stock relationship and generating the final input by changing `feature_cols`.

## 2. Train you model
* Before training, make sure to change the parameters in class `Args` and function `main`.

 ``` config
 adj_threshold = 0.1         # the threshold of the relations between stocks
 max_epochs = 60             # the number of training epochs
 epochs_eval = 10            # the number of training epochs per evaluation or test interval
 lr = 0.0002                 # learning rate of the model
 gamma = 0.3                 # gamma
 hidden_dim = 128            # hidden_dim
 num_heads = 8               # num_heads
 out_features = 32           # out_features
 batch_size = 512            # batch_size
 epochs_save_by = 60         # the number of training epochs of the saved model
 data_start = 0              # index of training start date
 data_middle = 653           # index of evaluation or test start date
 data_end = data_middle+251  # index of evaluation or test end date
 pre_data = '2021-12-31'     # save the last date of the training
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
      series = {CIKM '22},
      pages = {3584–3593},
	  numpages = {10},
      year={2022},
      doi = {10.1145/3511808.3557089}
    }
  ```