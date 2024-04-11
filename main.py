from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
import warnings
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

class Args:
    def __init__(self, gpu=0, subtask="regression"):
        # device
        self.gpu = str(gpu)
        self.device = 'cpu'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 60
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0002
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        # save model settings
        self.save_path = os.path.join(os.path.abspath('.'), "./data/model_saved/")
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)
        self.epochs_save_by = 60
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"


def fun_train_predict(data_start, data_middle, data_end, pre_data):
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dataset = AllGraphDataSampler(base_dir="./data/data_train_predict/", data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    val_dataset = AllGraphDataSampler(base_dir="./data/data_train_predict/", mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=mse_loss)
        if epoch % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=mse_loss)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss,
                                                                              eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))
        if (epoch + 1) % args.epochs_save_by == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))

    # predict
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
    model.load_state_dict(checkpoint['model'])
    data_code = os.listdir('./data/daily_stock')
    data_code = sorted(data_code)
    data_code_last = data_code[data_middle:data_end]
    df_score=pd.DataFrame()
    for i in tqdm(range(len(val_dataset))):
        df = pd.read_csv('./data/daily_stock/' + data_code_last[i], dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.train()
        logits = model(features, pos_adj, neg_adj)
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df_score=pd.concat([df_score,df])

        #df.to_csv('prediction/' + data_code_last[i], encoding='utf-8-sig', index=False)
    df_score.to_csv('./data/prediction/pred.csv')
    print(df_score)
    
if __name__ == "__main__":
    data_start = 20
    data_middle = 39
    data_end = data_middle+4
    pre_data = '2022-12-29'
    fun_train_predict(data_start, data_middle, data_end, pre_data)