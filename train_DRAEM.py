import torch
from DRAEM_module.data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from DRAEM_module.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from DRAEM_module.loss import FocalLoss, SSIM
import os
from tqdm import tqdm
import json

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(args):

    save_path = args['save_path']
    log_path = args['save_path'] + 'logs/'
    batch_size = args['batch_size']
    epochs = args['epoch']
    learning_rate = args['learning_rate']

    run_name = 'DRAEM'
    train_result_path = os.path.join(save_path, "train_result")
    log_path = os.path.join(train_result_path, "train_log")
    os.makedirs(train_result_path)
    os.makedirs(log_path)

    device_cuda = 'cuda'
    model = ReconstructiveSubNetwork(in_channels=1, out_channels=1)
    model.to(device=device_cuda)
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=2)
    model_seg.to(device=device_cuda)
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
                                    {"params": model.parameters(), "lr": learning_rate},
                                    {"params": model_seg.parameters(), "lr": learning_rate}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[epochs*0.8,epochs*0.9],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()
    
    dataset = MVTecDRAEMTrainDataset(args)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    early_stopping_count = 0
    early_stopping_patient = 10
    best_loss = 9999

    n_iter = 0
    
    print('Training ... ')
    
    for epoch in range(epochs):

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].to(device=device_cuda) #.cuda()
            aug_gray_batch = sample_batched["augmented_image"].to(device=device_cuda) #.cuda()
            anomaly_mask = sample_batched["anomaly_mask"].to(device=device_cuda) #.cuda()
            
            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            pbar.update(1) 
            pbar.set_postfix({"Loss": loss.item()})

            n_iter +=1

        scheduler.step()
        
        if best_loss > loss:
            best_loss = loss
            best_model = model
            best_model_seg = model_seg
            
            early_stopping_count = 0
            print(f'best model 👉 {epoch + 1} | loss : {best_loss}')
            
            torch.save(best_model.state_dict(), os.path.join(save_path, run_name+".pckl"))
            torch.save(best_model_seg.state_dict(), os.path.join(save_path, run_name+"_seg.pckl"))
        
        else :
            early_stopping_count += 1

        if early_stopping_count == early_stopping_patient:
            break

        
        log = dict()
        log['epoch'] = str(epoch)
        log['train_loss'] = str(round(float(loss), 4))

        with open(log_path + f'/{epoch}.json', 'w') as json_file:
            json.dump(log, json_file)

        print('Training Done !!')

