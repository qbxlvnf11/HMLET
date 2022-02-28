import numpy as np
import pandas as pd
import multiprocessing
import torch

import world
import utils

def BPR_train_original(dataset, recommend_model, loss_class, total_epochs, epoch, gum_temp, hard, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with utils.timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.bpr_batch_size + 1
    aver_loss = 0.
    
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.bpr_batch_size)):
        
        cri = bpr.stageOne(epoch, batch_users, batch_pos, batch_neg, gum_temp, hard)                                           
        
        aver_loss += cri

        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, total_epochs * int(len(users) / world.bpr_batch_size) + batch_i)
    aver_loss = aver_loss / total_batch

    return f"loss{aver_loss:.3f}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
def Test(dataset, model, epoch, gum_temp, hard, mode, w=None, multicore=0):

    u_batch_size = world.test_u_batch_size
    dataset: utils.BasicDataset
    
    # Mode
    if mode == 'valid':
      print('valid mode')
      testDict: dict = dataset.validDict
    elif mode == 'test':
      print('test mode')
      testDict: dict = dataset.testDict
    
    model = model.eval()
    max_K = max(world.topks)
    
    if multicore == 1:
        pool = multiprocessing.Pool(world.CORES)
    
    # Results
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
               
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            
        users_list = []
        rating_list = []
        groundTrue_list = []
        
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = model.getUsersRating(batch_users_gpu, gum_temp, 1, hard)         

            exclude_index = []
            exclude_items = []
            
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)          
            
        assert total_batch == len(users_list)
        
        X = zip(rating_list, groundTrue_list)
        
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
                
        scale = float(u_batch_size/len(users))
        
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        
        if multicore == 1:
            pool.close()
        
        return results
