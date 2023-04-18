import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
import torch.nn as nn
from PIL import Image
from configs import config
from tqdm import tqdm
from PIL import ImageFile
from sklearn.metrics import f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'





valid = pd.read_csv('valid.csv')
train = pd.read_csv('train.csv')
encoder = LabelEncoder()
train1 = train.copy()
valid1 = valid.copy()
train1['primary_label'] = encoder.fit_transform(train1['primary_label'])
valid1['primary_label'] = encoder.transform(valid1['primary_label'])

birds =sorted(train.primary_label.unique())

class BirdsSoundDataset(Dataset):
    def __init__(self, data, transformer=None, train=True):
        self.df = data
        self.audio_dir = data.filename.values
        self.transformer = transformer
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        mel_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        image = self._open_path(mel_path)
        if self.transformer == None:
            return image, label
        else:
            image = self.transformer(image)
            return image, label

    def _get_audio_sample_path(self, index):
        if self.train == True:
            fold = 'train_mel/'
            path = os.path.join(fold, self.audio_dir[index])
            return path.replace('.ogg','.png')
        else:
            fold = 'valid_mel/'
            path = os.path.join(fold, self.audio_dir[index])
            if 'ogg' in path:
                fold = 'train_mel/'
                path = os.path.join(fold, self.audio_dir[index])
                return path.replace('.ogg','.png')
            return path.replace('.mp3','.png')

    def _get_audio_sample_label(self, index):
        return self.df.iloc[index, 0]

    def _open_path(self,mel_path):
        image = Image.open(mel_path)
        return image

class BirdsSoundNNet(nn.Module):
    def __init__(self, num_classes=264):
        super(BirdsSoundNNet, self).__init__()
        self.model = efficientnet_b0(weights="DEFAULT")
        self.model.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

weights = EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms()


train_dataset = BirdsSoundDataset(train1,transformer=preprocess,train=True)
valid_dataset =BirdsSoundDataset(valid1,transformer=preprocess,train=False)

model_1 = BirdsSoundNNet()

def train_step(model,
               dataloader,
               loss_fn,
               optimizer):
    model.train()
    train_loss = 0
    train_f1_score = 0
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor), y.type(torch.LongTensor)
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_labels = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_f1_score += f1_score(y.cpu().detach().numpy(),y_pred_labels.cpu().detach().numpy(),average='macro')
    train_loss = train_loss/len(dataloader)
    train_f1_score = train_f1_score/len(dataloader)
    return train_loss,train_f1_score

def valid_step(model,dataloader,loss_fn,device = device):
    model.eval()
    test_loss,test_f1_score = 0,0
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):

            X,y = X.type(torch.FloatTensor),y.type(torch.LongTensor)
            X,y = X.to(device),y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_f1_score += f1_score(y.cpu().detach().numpy(),test_pred_labels.cpu().detach().numpy(),average='macro')
    test_loss = test_loss/len(dataloader)
    test_f1_score = test_f1_score/len(dataloader)
    # labels = np.vstack(labels)
    # preds = np.vstack(preds)
    # val_cmap = padded_cmap(pd.DataFrame(labels), pd.DataFrame(preds))
    return test_loss,test_f1_score


# def padded_cmap(solution, submission, padding_factor=5):
#     new_rows = []
#     for i in range(padding_factor):
#         new_rows.append([1 for i in range(len(solution.columns))])
#     new_rows = pd.DataFrame(new_rows)
#     new_rows.columns = solution.columns
#     padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
#     padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
#     score = average_precision_score(
#         padded_solution.values,
#         padded_submission.values,
#         average='macro',
#     )
#     return score








def train(model,train_dataloader,test_dataloader,optimizer,loss_fn,epochs = 5,device = device):
    history ={
        'train_loss':[],
        'train_f1_score':[],
        'valid_loss':[],
        'valid_f1_score':[],
        'cmap':[],
    }
    for epoch in tqdm(range(epochs)):
        train_loss,train_f1_score = train_step(model,train_dataloader,loss_fn,optimizer)
        valid_loss,valid_f1_score = valid_step(model,test_dataloader,loss_fn,device)
        print(f"Epoch: {epoch}| Train loss: {train_loss:.4f}| Train f1 score: {train_f1_score:.4f} | Valid loss: {valid_loss:.4f}| Valid f1 score: {valid_f1_score:.4f}")
        #print(f'f1_score: {cmap_sore}')
        history['train_loss'].append(train_loss)
        history['train_f1_score'].append(train_f1_score)
        history['valid_loss'].append(valid_loss)
        history['valid_f1_score'].append(valid_f1_score)
        #validation_epoch_end(test_loss,epoch)
    return history

train_dataloader = DataLoader(train_dataset,batch_size=100,num_workers=2,drop_last=True,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=100,num_workers=2)

model_1.to(device)
optimizer = torch.optim.Adam(model_1.parameters(),lr = config.lr)
loss_fn = nn.CrossEntropyLoss()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train(model_1, train_dataloader, valid_dataloader, optimizer, loss_fn,epochs=3)

    # torch.save(model_1.state_dict(), 'BirdsSoundClassification')
    # torch.save(model_1.state_dict(), 'model100epochsWieghts')
    # print(model_1.to('cpu').state_dict())
    # torch.save(model_1.to('cpu').state_dict(),'model1_100_epochs_weights')

    

