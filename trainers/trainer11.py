import sys
sys.path.append("..")
from models.model11 import *
from utils.utils import *
from utils.data import DataLoad
import config

class Train():
    def __init__(self):
        self.IMAGE_WIDTH = config.IMAGE_WIDTH
        self.IMAGE_HEIGHT = config.IMAGE_HEIGHT
        self.train_val_years = config.train_val_years
        self.test_years = config.test_years
        self.target = config.trainer11['target']
        self.path = config.trainer11['path']
        self.params = config.trainer11

    def dataStep(self):
        data = DataLoad(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.train_val_years, self.test_years, self.target, self.path, self.params['batch_size'])
        [self.train_loader, self.val_loader, self.test_loader] = data.main()
    
    def model(self):
        model = CNN()
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.device_ids=range(torch.cuda.device_count())
        model.cuda(device=self.device_ids[3])
        self.model = model 
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.params['init_lr'])

        self.early_stopping = EarlyStopping(patience = self.params['early_stopping_patience'],delta = self.params['early_stopping_delta'], path= self.params['early_stopping_path'],verbose=True)

    def train(self):
        self.loss_count = []
        epochs = self.params['epoch']
        self.global_loss_train = []
        self.global_loss_test = []
        self.global_loss_val = []

        lr = []

        for epoch in range(epochs):
            
            for i,(x,y) in enumerate(self.train_loader):
                batch_x = Variable(x.cuda(device=self.device_ids[3]))
                batch_y = Variable(y.cuda(device=self.device_ids[3]))
                out = self.model(batch_x.float())
                loss = self.loss_func(out,batch_y.squeeze().long())
                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()
                if i%20 == 0:
                    temp = loss.cpu()
                    self.loss_count.append(temp.detach().numpy())
                    print('epoch:', format(epoch+1),f'iteration: {i+1}:\t','loss:', loss.item())
                    
            
            loss_val_epoch = [] 
            for x,y in self.val_loader:
                batch_x = Variable(x.cuda(device=self.device_ids[3]))
                batch_y = Variable(y.cuda(device=self.device_ids[3]))
                prediction = self.model(batch_x)
                loss = self.loss_func(prediction, batch_y.squeeze().long())
                loss_val_epoch.append(loss.cpu().detach().numpy())
                
            loss_val = np.mean(loss_val_epoch)
            self.global_loss_val.append(loss_val)

            loss_train = self.loss_count[-1]
            self.global_loss_train.append(loss_train)
            
            self.early_stopping(loss_val, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            print('----------------epoch '+str(epoch+1)+' end---------------------')
        self.lr = lr
        self.model = torch.load(self.params['early_stopping_path'])
        
    
    def evaluate(self):
        self.model = torch.load(self.params['early_stopping_path'])
        
        #########################################################################
        plt.figure(figsize=(10,5))
        plt.title('PyTorch_CNN_Loss')
        plt.plot(self.loss_count,label='Loss')
        plt.plot(pd.DataFrame(np.array(self.loss_count)).rolling(20,1).mean(),label='MA Loss')
        plt.legend()
        plt.savefig('./reports/figures/trainer11_CNN_classification_loss.png')
        plt.show()
        
        #########################################################################
        epochs = range(1, len(self.global_loss_train) + 1)
        minposs = self.global_loss_val.index(self.early_stopping.val_loss_min.tolist())+1
        fig,ax=plt.subplots(1,2,figsize=(10,5))

        ax[0].plot(epochs, self.global_loss_train, '.', label='Train')
        ax[0].plot(epochs, self.global_loss_train,'r')
        ax[0].axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax[0].legend()
        ax[0].set_title('train_loss')
        ax[1].plot(epochs, self.global_loss_val, '.', label='Val')
        ax[1].plot(epochs, self.global_loss_val,'r')
        ax[1].axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax[1].legend()
        ax[1].set_title('val_loss')
        plt.savefig('./reports/figures/trainer11_CNN_classification_train_val_loss.png')
        plt.show()

        data = DataLoad(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.train_val_years, config.test_years, self.params['target'], self.params['path'], self.params['batch_size'])
        [x_train, x_val, y_train, y_val, images_test, label_test] = data.processing_data()
        
        #########################################################################
        prediction = []
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        device_ids=range(torch.cuda.device_count())
        global_loss_test = []
        with torch.no_grad():
            for images,labels in self.test_loader:
                test_x = Variable(images.cuda(device=device_ids[3]))
                test_y = Variable(labels.cuda(device=device_ids[3]))
                prediction_test = self.model(test_x)
                prediction.extend(torch.max(prediction_test,1)[1].cpu().numpy().tolist())
                loss_test = self.loss_func(prediction_test, test_y.squeeze().long())
                global_loss_test.append(loss_test.cpu().detach().numpy())
        label_test['pred'] = prediction
        cm = confusion_matrix(label_test[self.params['target']], label_test['pred'])
        print("Test Accuracy", (cm[0][0]+cm[1][1])/np.sum(cm))
        print('Test Loss', np.mean(global_loss_test))
        print("Test Pearsonr:", scipy.stats.pearsonr(label_test[self.params['target']], label_test['pred'])[0])
        print("Test Spearmanr:", scipy.stats.spearmanr(label_test[self.params['target']], label_test['pred'])[0])
        
        #########################################################################
        prediction = []
        global_loss_val = []
        with torch.no_grad():
            for images,labels in self.val_loader:
                val_x = Variable(images.cuda(device=device_ids[3]))
                val_y = Variable(labels.cuda(device=device_ids[3]))
                prediction_val = self.model(val_x)
                prediction.extend(torch.max(prediction_val,1)[1].cpu().numpy().tolist())
                loss_val = self.loss_func(prediction_val, val_y.squeeze().long())
                global_loss_val.append(loss_val.cpu().detach().numpy())
        y_val['pred'] = prediction
        cm = confusion_matrix(y_val[self.params['target']], y_val['pred'])
        print("Val Accuracy", (cm[0][0]+cm[1][1])/np.sum(cm))
        print('Val Loss', np.mean(global_loss_val))
        
        #########################################################################
        
        
    def main(self):
        print(f"{'DATA':-^60}")
        self.dataStep()
        print(f"{'Init Model':-^60}")
        self.model()
        print(f"{'Train Model':-^60}")
        self.train()
        print(f"{'Evaluate Model':-^60}")
        self.evaluate()