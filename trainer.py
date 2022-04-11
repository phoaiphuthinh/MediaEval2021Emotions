import os
import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from model import Model
from loss import WeightedFocalLoss, CoteachingLoss
from transforms import get_transforms

class Trainer(object):
    def __init__(self, args, train_dataset=None, val_df = None, test_df = None, path_audio=None) -> None:
        self.args = args
        self.names = args.name
        self.train_dataset = train_dataset

        self.val_df = val_df
        self.test_df = test_df
        self.path_audio = path_audio

        self.transform = get_transforms(False, args.size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = list(map(lambda name: Model(name, 56).to(self.device), self.names))
        self.optimizers = list(map(lambda m: optim.Adam(m.parameters(), lr=0.001, amsgrad=True), self.models))
        self.schedulers = list(map(lambda op: optim.lr_scheduler.ReduceLROnPlateau(op, patience=5, verbose=True), self.optimizers))

        self.base_loss = WeightedFocalLoss(56, gamma=2)
        self.coteaching_loss = CoteachingLoss(self.base_loss, args.forget_rate, self.device)

    def eval_samples(self, fn):
        name = int(fn)

        x = torch.zeros(self.args.chunk_size, 1, 96, self.args.size)
        path = os.path.join(self.args.data_dir, str(self.path_audio[name]), str(name)+'.npy')
        mel = np.load(path)
        mel_len = mel.shape[1]

        chunk = (mel_len - self.args.cut_size) // self.args.chunk_size

        for i in range(self.args.chunk_size):
            offset = i * chunk
            cut_mel = mel[:, offset:(offset + self.args.cut_size)]
            x[i] = self.transform(cut_mel)
        return x

    def train(self):
        
        train_loader_1 = DataLoader(self.train, self.args.batch_size, shuffle=True, drop_last=True)
        train_loader_2 = DataLoader(self.train, self.args.batch_size, shuffle=True, drop_last=True)
        
        best = [0 for _ in range(len(self.models))]
        count_stop = 0
        for ep in range(self.args.epoch):
            print('Epoch: ', ep)
            total_loss = 0
            for it1, it2 in zip(train_loader_1, train_loader_2):
                alpha = 1
                mixup_vals = np.random.beta(alpha, alpha, it1[0].shape[0])
        
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
                mels = (lam * it1[0]) + ((1 - lam) * it2[0])
                #wavs = (lam * i1[1]) + ((1 - lam) * i2[1])
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
                labels = (lam * it1[1]) + ((1 - lam) * it2[1])
                # mixup ends ----------
                mels = mels.to(self.device, non_blocking=False)
                labels = labels.to(self.device, non_blocking=False)

                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    for model in self.models:
                        model.train()
                    logits = list(map(lambda model: model(mels), self.models))
                    assert len(logits) <= 2
                    losses = self.coteaching_loss(logits, labels)
                    for loss in losses:
                        loss.backward()
                    for optimizer in self.optimizers:
                        optimizer.step()
                    total_loss += sum(map(lambda loss: loss.detach().cpu().numpy()))
            
            results = self.evaluate('dev')

            print('Evaluated!')
            print(results)

            for i in range(len(best)):
                if results['pr_auc'][i] > best[i]:
                    count_stop = 0
                    self.save_model(self.names[i], self.models[i])
                    print('Saved model {}...'.format(self.names[i]))

            count_stop += 1
            if count_stop >= 10:
                break
            
            for i in range(len(self.schedulers)):
                self.schedulers[i].step(results['loss'][i])


    def evaluate(self, mode):
        all_logits = [[] for _ in range(len(self.models))]
        all_labels = []

        losses = [0 for _ in range(len(self.models))]

        if mode == 'dev':
            df = self.val_df
        else:
            df = self.test_df

        for fn, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            mel = self.get_tensor(fn)
            ground_truth = row.values
            
            mel = mel.to(self.device)
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                for model in self.models:
                    model.eval()
                logits = list(map(lambda model: model(mel), self.models))
                predictions = list(map(lambda logit: np.array(logit.detach().cpu()).mean(axis=0), logits))
                avg = list(map(lambda logit: torch.mean(logit, dim=0).unsqueeze(0), logits))
                loss = list(map(lambda logit: self.coteaching_loss(logit, torch.tensor([ground_truth]).type(torch.cuda.FloatTensor)), avg))
                loss = list(map(lambda l: l.mean(dim=0).mean(dim=0).detach().cpu().numpy(), loss))

                for i in range(len(all_logits)):
                    all_logits[i].append(predictions[i])
                    losses[i] += loss[i]

                all_labels.append(ground_truth)
        
        all_logits = list(map(lambda logit: torch.sigmoid(np.concatenate(logit, axis=0)), all_logits))
        all_labels  = np.concatenate(all_labels,  axis=0)


        pr_auc = list(map(lambda logit: average_precision_score(all_labels, logit, average='macro'), all_logits))
        roc_auc = list(map(lambda logit: roc_auc_score(all_labels, logit, average='macro'), all_logits))
        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "total_loss": sum(losses),
            "loss": losses
        }

    def save_model(self, name, model_to_save):
        path = os.path.join(self.args.model_dir, 'checkpoints', name)
        torch.save(model_to_save.state_dict(), path)

    def load_models(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            for i in range(len(self.models)):
                path = os.path.join(self.args.model_dir, 'checkpoints', self.names[i])
                self.models[i].load_state_dict(torch.load(path))
        except Exception:
            raise Exception("Some model files might be missing...")