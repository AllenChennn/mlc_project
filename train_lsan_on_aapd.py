import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
import utils.lsan_util as util
from models.lsan import StructuredSelfAttention
from tqdm import tqdm

## 加载配置
config = util.read_config("files/lsan/config.yml")
if config.GPU:
    torch.cuda.set_device(0)
batch_size = config.batch_size

## 导入数据
def load_data():
    # 修改路径
    X_tst = np.load("data/AAPD/X_test.npy")
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    Y_tst = np.load("data/AAPD/y_test.npy")

    return (X_trn, Y_trn), (X_tst, Y_tst)

train_data, test_data = load_data()

## 数据生成器

train_data = data_utils.TensorDataset(torch.from_numpy(train_data[0]).type(torch.LongTensor),
                                          torch.from_numpy(train_data[1]).type(torch.LongTensor))
test_data = data_utils.TensorDataset(torch.from_numpy(test_data[0]).type(torch.LongTensor),
                                        torch.from_numpy(test_data[1]).type(torch.LongTensor))
train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)


## 定义模型
word_embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt').idx_to_vec.asnumpy()
label_embed = np.load("data/AAPD/label_embed.npy")
word_embed = torch.from_numpy(word_embed).float()
label_embed = torch.from_numpy(label_embed).float()

label_num = label_embed.shape[0]

model = StructuredSelfAttention(
    batch_size=batch_size, 
    lstm_hid_dim=config['lstm_hidden_dimension'],
    d_a=config["d_a"], 
    n_classes=label_num, 
    label_embed=label_embed,
    embeddings=word_embed
)

loss = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

model_file = 'output/lsan_aapd.best_model'

class Trainer(object):
    def __init__(self, model, criterion, optimizer, train_dataset, test_dataset, epochs, use_cuda = False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.n_epoch = 1
        self.best_eval_score = .0

        if self.use_cuda:
            self.model = self.model.cuda()
    
    def run(self):
        for i in range(self.epochs):
            self.train()
            self.evaluation()

            self.n_epoch += 1
    
    def train(self):
        self.model.train()

        print("Running EPOCH",self.n_epoch)
        train_loss = []
        prec_k = []
        ndcg_k = []
        for i, (x, y) in enumerate(tqdm(self.train_dataset)):
            self.optimizer.zero_grad()
            # x, y = train[0].cuda(), train[1].cuda()
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            # 前向传播
            out = self.model(x)
            loss = self.criterion(out, y.float()) / x.shape[0]
            # 反向传播，更新参数
            loss.backward()
            
            self.optimizer.step()
            # 计算指标
            labels_cpu = y.data.cpu().float()
            pred_cpu = out.data.cpu()
            prec = util.precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg = util.Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)

            prec_k.append(prec)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        
        # 打印日志
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (self.n_epoch, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        
    
    def evaluation(self):
        self.model.eval()

        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        with torch.no_grad(): # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播
            for i, (x, y) in enumerate(self.test_dataset):
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                out = self.model(x)
                loss = self.criterion(out, y.float()) / x.shape[0]

                labels_cpu = y.data.cpu().float()
                pred_cpu = out.data.cpu()

                prec = util.precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
                ndcg = util.Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)

                test_acc_k.append(prec)
                test_ndcg_k.append(ndcg)
                test_loss.append(float(loss))

        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)
        print("epoch %2d test end : avg_loss = %.4f" % (self.n_epoch, avg_test_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        test_prec[0], test_prec[2], test_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))
        
        if test_prec[0] > self.best_eval_score:
            self.model.save(model_file)


if __name__=='__main__':
    Trainer(
        model, 
        criterion=loss,
        optimizer=opt,
        train_dataset=train_loader, 
        test_dataset=test_loader,
        epochs=10,
        use_cuda=config.GPU
    ).run()
    
else:
    model.load_state_dict(torch.load(model_file))