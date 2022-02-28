import os
from os.path import join
import multiprocessing
import torch

from parse import parse_args

# Model
from models import HMLET_END
from models import HMLET_FRONT
from models import HMLET_MIDDLE
from models import HMLET_ALL

args = parse_args()

dataset = args.dataset
model_name = args.model

MODELS = {
  "HMLET_End": HMLET_END.HMLET_End,
  "HMLET_Front": HMLET_FRONT.HMLET_Front,
  "HMLET_Middle": HMLET_MIDDLE.HMLET_Middle,
  "HMLET_All": HMLET_ALL.HMLET_All
}

# Model & Train Param
EPOCHS = args.epochs
SEED = args.seed
pretrain = True if args.pretrain else False
load_epoch = args.load_epoch
topks = eval(args.topks)
a_n_fold = args.a_n_fold
tensorboard = True if args.tensorboard else False

bpr_batch_size = args.bpr_batch
test_u_batch_size = args.testbatch
lr = args.lr
decay = args.decay

config = {}
config['a_split'] = args.a_split
config['embedding_dim'] = args.embedding_dim
config['activation_function'] = args.non_linear_acti
config['graph_dropout']  = args.graph_dropout
config['graph_keep_prob']  = args.graph_keep_prob
config['gating_mlp_dims'] = [512, 128, 64, 2]
config['gating_dropout_prob']  = args.gating_dropout_prob

print('='*30)
print('Model:', model_name)
print('Model config:', config)
print('Dataset:', dataset)
print("EPOCHS:", EPOCHS)
print("Pretrain:", pretrain)
print("BPR batch size:", bpr_batch_size)
print("Test batch size:", test_u_batch_size)
print("Test topks:", topks)
print("N fold:", a_n_fold)
print("Tensorboard:", tensorboard)
print('='*30)


# Gumbel-Softmax Param
ori_temp = args.ori_temp
min_temp = args.min_temp
gum_temp_decay = args.gum_temp_decay
epoch_temp_decay = args.epoch_temp_decay
train_hard = False
test_hard = True

# PATH
ROOT_PATH = "./"
DATA_PATH = join(ROOT_PATH, 'data', args.dataset)
SAVE_FILE_PATH = join(ROOT_PATH, args.save_checkpoints_path, model_name, dataset)
LOAD_FILE_PATH = join(ROOT_PATH, args.save_checkpoints_path, model_name, dataset, args.pretrained_checkpoint_name)
BOARD_PATH = join(ROOT_PATH, 'tensorboard')

print('='*30)
print('DATA PATH:', DATA_PATH)
print('SAVE FILE PATH:', SAVE_FILE_PATH)
print('LOAD FILE PATH:', LOAD_FILE_PATH)
print('BOARD PATH:', BOARD_PATH)
print('='*30)

# Making folder
os.makedirs(SAVE_FILE_PATH, exist_ok=True)
os.makedirs(BOARD_PATH, exist_ok=True)
   
# GPU
print('='*30)
print('Cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
  GPU_NUM = args.gpu_num
  print('GPU_NUM:', GPU_NUM)
  device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
  torch.cuda.set_device(device)
print('='*30)


# Multi-processing 
multicore = args.multicore
CORES = multiprocessing.cpu_count() // 2
print('='*30)
print("Multicore:", multicore)
print("CORES:", CORES)
print('='*30)

# Excel results dict
excel_results_valid = {}
excel_results_valid['Model'] = []
excel_results_valid['Dataset'] = []
excel_results_valid['Epochs'] = []
excel_results_valid['Precision'] = []
excel_results_valid['Recall(HR)'] = []
excel_results_valid['Ndcg'] = []

excel_results_test = {}
excel_results_test['Model'] = []
excel_results_test['Dataset'] = []
excel_results_test['Epochs'] = []
excel_results_test['Precision'] = []
excel_results_test['Recall(HR)'] = []
excel_results_test['Ndcg'] = []
