import world
import dataloader
import procedure
import utils

import pandas as pd
import math
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import os
from os.path import join
from torch import nn

utils.set_seed(world.SEED)

dataset = dataloader.Loader(world.DATA_PATH)

model = world.MODELS[world.model_name](world.config, dataset)
model = model.to(world.device)

# Pretrain
if world.pretrain:
    try:
        pretrained_file = world.LOAD_FILE_PATH
        model.load_state_dict(torch.load(pretrained_file, map_location=world.device))
        print(f"loaded model weights from {pretrained_file}")
    except FileNotFoundError:
        print(f"{pretrained_file} not exists, start from beginning")

bpr = utils.BPRLoss(model)

# Tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-"))
                                    )
else:
    w = None
    print("not enable tensorflowboard")

try:
    start_epoch = world.load_epoch
    gum_temp = world.ori_temp
    for epoch in range(start_epoch, world.EPOCHS+1):
        start = time.time()
        
        print('Train', epoch, '='*30)
        output_information = procedure.BPR_train_original(dataset, model, bpr, world.EPOCHS, epoch, gum_temp, hard=world.train_hard, w=w)
        print(f'EPOCH[{epoch}/{world.EPOCHS}] {output_information}')
        
        end = time.time()
        print('train time:', end-start)

        print("model save...")
        torch.save(model.state_dict(), world.SAVE_FILE_PATH+'/'+world.model_name+'_'+world.dataset+'_'+str(epoch)+".pth.tar")
        
        if epoch > 300 and epoch % world.epoch_temp_decay == 0:
            # Temp decay
            gum_temp = world.ori_temp * math.exp(-world.gum_temp_decay*(epoch-300))
            gum_temp = max(gum_temp, world.min_temp)
            
        if epoch % 10 == 0:

            print('Valid', '='*50)
            valid_results = procedure.Test(dataset, model, epoch, gum_temp, hard=world.test_hard, mode='valid', w=w, multicore=world.multicore)   
            print('valid_results:', valid_results)     
            
            print('Test', '='*50)
            test_results = procedure.Test(dataset, model, epoch, gum_temp, hard=world.test_hard, mode='test', w=w, multicore=world.multicore)
            print('test_results:', test_results) 
            
finally:
    if world.tensorboard:
        w.close()
