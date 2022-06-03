import os
from random import shuffle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
torch.manual_seed(44)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset import Dataset_ESD, collate_fn
from torch.utils.data import DataLoader

from VC_model import VoiceConversionModel

import wandb
import time
from tqdm import tqdm

from utils import check_recon_mel, check_tsne, makedirs


class Train():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.wandb_login = config['Train']['wandb_login']

        """ HyperParameters """
        self.batch_size = config['Train']['batch_size']
        self.lr = config['Train']['learning_rate']
        self.weight_decay = config['Train']['weight_decay']
        self.num_workers = config['Train']['num_workers']

        self.lambda_spk = config['Train']['lambda_spk']
        self.lambda_vq = config['Train']['lambda_vq']

        step_size = config['Train']['scheduler_size']
        gamma = config['Train']['scheduler_gamma']

        stats_file = "../ESD_preprocessed/mel_stats.npy"
        self.mel_stats = np.load(stats_file).astype(np.float64)

        """ Model, Optimizer """
        self.model = VoiceConversionModel(config).to(self.device)

        print("Autoencoder: {}".format(self.get_n_params(self.model)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)



        """ Dataset """
        self.crop_mel = config['Loader']["mel_length"]

        self.dataset_train = Dataset_ESD(self.crop_mel, 'train')
        self.dataset_eval = Dataset_ESD(self.crop_mel, 'eval')

        """ Path """
        # Save Model Path: "./assets/ts/model/"
        # Save Fig: "./assets/ts/figs/"
        self.dir_path = config['Result']['asset_dir_path']
        self.save_step = config['Train']['save_for_step']
        


    def fit(self, tot_epoch):
        """ Training DataLoader """
        self.train_loader = DataLoader(
            self.dataset_train, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers
        )

        """ Evaluation DataLoader """
        self.eval_loader = DataLoader(
            self.dataset_eval, batch_size=self.batch_size, drop_last=True,
            shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
        )

        self.cur_step = 0
        self.outer_pbar = tqdm(total=tot_epoch, desc=f"Training... >>>> Total {tot_epoch} Epochs", position=0)

        ts = time.time()
        self.asset_path = self.dir_path + str(ts)
        makedirs(self.asset_path)


        """ Make Funny Training!! """
        for epo in range(tot_epoch):
            self.training_step()
            self.validation_step()


    def step(self, batch):
        mel, spk_id, emo_id = list(map(lambda x: x.to(self.device), batch))
        
        # forward
        recon_mel, post_mel, vq_loss, spk_loss, _, _, _ = self.model(mel, spk_id, emo_id)

        # calculate loss
        mel_loss = self.spec_loss(mel, recon_mel)
        post_loss = self.spec_loss(mel, post_mel)
        
        tot_loss = mel_loss + post_loss + vq_loss * self.lambda_vq + spk_loss * self.lambda_spk

        return tot_loss, mel_loss, post_loss, vq_loss, spk_loss


    def training_step(self):
        for batch in tqdm(self.train_loader):
            """ Training step """
            # zero grad
            self.optimizer.zero_grad()
            
            # forward & calculate total loss
            tot_loss, mel_loss, post_loss, vq_loss, spk_loss = self.step(batch)

            # backwarding
            tot_loss.backward()

            # optimize
            self.optimizer.step()
            self.scheduler.step()


            """ end """
            loss_dict = {
                "Total Loss": tot_loss.item(), "Mel Loss": mel_loss.item(), "Post Loss": post_loss.item(),
                "VQ Loss": vq_loss.item(), "SPK Loss": spk_loss.item()
            }
            
            self.training_step_end(loss_dict)
            

    def training_step_end(self, loss_dict):
        self.cur_step += 1

        # Update current learning rate in the dictionary
        loss_dict.update( {"lr": self.optimizer.param_groups[0]['lr']} )
        self.outer_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            wandb.log(loss_dict)

        if self.cur_step % self.save_step == 0:
            save_path = os.path.join(self.asset_path, 'checkpoint_{}.pth.tar'.format(self.cur_step))
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            torch.save(save_dict, save_path)
            print("save model at step {} ...".format(self.cur_step))


    def step_eval(self, batch):
        mel, spk_id, emo_id = list(map(lambda x: x.to(self.device), batch))
        with torch.no_grad():
            _, pred_mel, vq_loss, spk_loss, _, _, _ = self.model(mel, spk_id, emo_id)
            
            # calculate loss
            mel_loss = self.spec_loss(mel, pred_mel)
            
        return pred_mel, mel_loss, vq_loss, spk_loss


    def validation_step(self):
        with torch.no_grad():
            eval_pbar = tqdm(self.eval_loader, desc="Validation...")

            for batch in eval_pbar:
                recon_mel, mel_loss, vq_loss, spk_loss = self.step_eval(batch)

                """ log """
                loss_dict = {
                    "Post Val Loss": mel_loss.item(),
                    "Spk Val Loss": spk_loss.item(),
                    "Q Val Loss": vq_loss.item(),
                }

                eval_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            wandb.log(loss_dict)

        check_recon_mel(self.denormalize(recon_mel[0].to('cpu').detach().numpy(), *self.mel_stats).T, 
            self.asset_path, self.outer_pbar.n, mode='recon')
        check_recon_mel(self.denormalize(batch[0][0].to('cpu').detach().numpy(), *self.mel_stats).T, 
            self.asset_path, self.outer_pbar.n, mode='GT')

        with open(self.asset_path + "/config.yaml", 'w') as file:
            documents = yaml.dump(self.config, file)

        self.outer_pbar.update()
        eval_pbar.close()


    def spec_loss(self, mel, pred_mel):
        return F.l1_loss(mel, pred_mel)

    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def denormalize(self, mel, mean, std):
        return mean + std * mel


        

    


import argparse
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, 
        default=400
    )
    parser.add_argument('--gpu_visible_devices', type=str, default='0, 1, 2, 3, 4, 5')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("../configs/config_VC.yaml", "r"), Loader=yaml.FullLoader
    )

    wandb_login = config['Train']['wandb_login']
    lr = config['Train']['learning_rate']

    args = argument_parse()

    if wandb_login:
        wandb.login()
        wandb_name = "Recon_VC"
        wandb.init(project='Recon_Emotion_VC', name=wandb_name)

    trainer = Train(config)
    trainer.fit(args.epochs)
