import argparse

def parse_args():
    parser = argparse.ArgumentParser()
	
    # Model
    parser.add_argument('--model', type=str, default="HMLET_End",
							choices={"HMLET_End", "HMLET_Middle", "HMLET_Front", "HMLET_All"},
							help="model type")
    parser.add_argument('--embedding_dim', type=int, default=512,
							help="the embedding size")
    parser.add_argument('--non_linear_acti', type=str, default="elu",
              choices={"relu", "leaky-relu", "elu"},
              help="activation function to use in non-linear aggregation")
    parser.add_argument('--graph_dropout', type=int, default=1,
							help="using the dropout or not")
    parser.add_argument('--graph_keep_prob', type=float, default=0.6,
							help="1 - dropout rate")
    parser.add_argument('--gating_dropout_prob', type=float, default=0.2,
							help="dropout rate of gating networks")
						
    # Dataset
    parser.add_argument('--dataset', type=str, default='gowalla',
							choices={"gowalla", "yelp2018", "amazon-book"},
							help="dataset")
    parser.add_argument('--bpr_batch', type=int, default=2048,
							help="the batch size for bpr loss training procedure")	

    # Gumbel-Softmax
    parser.add_argument('--ori_temp', type=float, default=0.7,
							help='start temperature')
    parser.add_argument('--min_temp', type=float, default=0.01,
							help='min temperature')
    parser.add_argument('--gum_temp_decay', type=float, default=0.005,
							help='value of temperature decay')
    parser.add_argument('--epoch_temp_decay', type=int, default=1,
							help='epoch to apply temperature decay')
						
    # Train
    parser.add_argument('--epochs', type=int, default=1000,
							help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001,
							help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
							help="the weight decay for l2 normalizaton")
								
    # Test
    parser.add_argument('--topks', nargs='?', default="[10,20,30,40,50]",
							help="@k test list")
    parser.add_argument('--testbatch', type=int, default=100,
							help="the batch size of users for testing")
    parser.add_argument('--a_split', default=0,
							help="split large adj matrix or not")
    parser.add_argument('--a_n_fold', type=int, default=100,
							help="the fold num used to split large adj matrix")
						
    # Util
    parser.add_argument('--pretrain', type=int, default=0,
							help='using pretrained weight or not')
    parser.add_argument('--pretrained_checkpoint_name', type=str, default='',
							help='file name of pretrained model')
    parser.add_argument('--load_epoch', type=int, default=1,
							help='epoch of pretrained model')
    parser.add_argument('--seed', type=int, default=2020,
							help='random seed')
    parser.add_argument('--multicore', type=int, default=0,
							help='using multiprocessing or not')
    parser.add_argument('--gpu_num', type=int, default=0,
							help='gpu number')     
    parser.add_argument('--save_checkpoints_path', type=str, default="checkpoints",
							help="path to save weights")
    parser.add_argument('--tensorboard', type=int, default=0,
							help="enable tensorboard")
						
    return parser.parse_args()
