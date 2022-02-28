import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

from .gating.gating_network import Gating_Net

class HMLET_End(nn.Module):
	def __init__(self, 
					config:dict, 
					dataset:BasicDataset):
		super(HMLET_End, self).__init__()
		self.config = config
		self.dataset : dataloader.BasicDataset = dataset
		self.__init_model()

	def __init_model(self):

		# Num of users & items
		self.num_users = self.dataset.n_users
		self.num_items = self.dataset.m_items
		print(f'user: {self.num_users}, items: {self.num_items}')

		# Embeddings
		self.embedding_dim = self.config['embedding_dim']
		self.embedding_user = torch.nn.Embedding(
			num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
		self.embedding_item = torch.nn.Embedding(
			num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

		# Layer
		self.n_layers = 4

		# Graph Dropout
		self.dropout = self.config['graph_dropout']
		self.keep_prob = self.config['graph_keep_prob']

    	# Gating Net with Gumbel-Softmax
		self.gating_dropout_prob = self.config['gating_dropout_prob']
		self.gating_mlp_dims = self.config['gating_mlp_dims']
		self.gating_net1 = Gating_Net(3, embedding_dim=self.embedding_dim, mlp_dims=self.gating_mlp_dims, dropout_p=self.gating_dropout_prob).to(world.device)
		self.gating_net2 = Gating_Net(4, embedding_dim=self.embedding_dim, mlp_dims=self.gating_mlp_dims, dropout_p=self.gating_dropout_prob).to(world.device)

		# Split
		self.A_split = self.config['a_split']
		
		# Normal distribution initilizer
		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)      
   
		# Activation function
		selected_activation_function = self.config['activation_function']
		if selected_activation_function == 'relu':
			self.r = nn.ReLU()
			self.activation_function = self.r
		if selected_activation_function == 'leaky-relu':
			self.leaky = nn.LeakyReLU(0.1)
			self.activation_function = self.leaky
		elif selected_activation_function == 'elu':
			self.elu = nn.ELU()
			self.activation_function = self.elu
		print('activation_function:',self.activation_function)
		
		# Train graphs
		self.g_train = self.dataset.getSparseGraph()

		# Freezing until training sufficiently user and item embedding
		self.__gating_freeze(self.gating_net1, False)
		self.__gating_freeze(self.gating_net2, False)

	def __gating_freeze(self, model, freeze_flag):
		for name, child in model.named_children():
			for param in child.parameters():
				param.requires_grad = freeze_flag

	def __choosing_one(self, features, gumbel_out):
		feature = torch.sum(torch.mul(features, gumbel_out), dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
		return feature

	def __dropout_x(self, x, keep_prob):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index]/keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g

	def __dropout(self, keep_prob):
		if self.A_split:   
			graph = []
			for g in self.Graph:
				graph.append(self.__dropout_x(g, keep_prob))
		else:
			graph = self.__dropout_x(self.Graph, keep_prob)
		return graph

	def computer(self, gum_temp, div_noise, hard):     
		
		self.Graph = self.g_train   
		if self.dropout:
			if self.training:
				g_droped = self.__dropout(self.keep_prob)
			else:
				g_droped = self.Graph        
		else:
			g_droped = self.Graph
    
    
		# Init users & items embeddings  
		users_emb = self.embedding_user.weight
		items_emb = self.embedding_item.weight
      
      
		## Layer 0
		all_emb_0 = torch.cat([users_emb, items_emb])
		
		# Residual embeddings
		embs = [all_emb_0]
		
   
		## Layer 1
		all_emb_lin_1 = torch.sparse.mm(g_droped, all_emb_0)
		
		# Residual embeddings	
		embs.append(all_emb_lin_1)
		
   
		## layer 2
		all_emb_lin_2 = torch.sparse.mm(g_droped, all_emb_lin_1)
		
		# Residual embeddings
		embs.append(all_emb_lin_2)
		
   
		## layer 3
		all_emb_lin_3 = torch.sparse.mm(g_droped, all_emb_lin_2)
		all_emb_non_1 = self.activation_function(torch.sparse.mm(g_droped, all_emb_0))
		
		# Gating
		stack_embedding_1 = torch.stack([all_emb_lin_3, all_emb_non_1],dim=1)
		concat_embeddings_1 = torch.cat((all_emb_lin_3, all_emb_non_1),-1)

		gumbel_out_1 = self.gating_net1(concat_embeddings_1, gum_temp, hard, div_noise)
		embedding_1 = self.__choosing_one(stack_embedding_1, gumbel_out_1)
		
		# Residual embeddings
		embs.append(embedding_1)
	
  	
		# layer 4
		all_emb_lin_4 = torch.sparse.mm(g_droped, embedding_1)
		all_emb_non_2 = self.activation_function(torch.sparse.mm(g_droped, embedding_1))
    		
		# Gating
		stack_embedding_2 = torch.stack([all_emb_lin_4, all_emb_non_2],dim=1)
		concat_embeddings_2 = torch.cat((all_emb_lin_4, all_emb_non_2),-1)

		gumbel_out_2 = self.gating_net2(concat_embeddings_2, gum_temp, hard, div_noise)
		embedding_2 = self.__choosing_one(stack_embedding_2, gumbel_out_2)

		# Residual embeddings  		
		embs.append(embedding_2)


		## Stack & mean residual embeddings
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
   
		users, items = torch.split(light_out, [self.num_users, self.num_items])

		return users, items
		
	def getUsersRating(self, users, gum_temp, div_noise, hard):
		all_users, all_items = self.computer(gum_temp, div_noise, hard)
		
		users_emb = all_users[users.long()]
		items_emb = all_items

		rating = self.activation_function(torch.matmul(users_emb, items_emb.t()))

		return rating

	def getEmbedding(self, users, pos_items, neg_items, gum_temp, div_noise, hard):
		all_users, all_items = self.computer(gum_temp, div_noise, hard)
		
		users_emb = all_users[users]
		pos_emb = all_items[pos_items]
		neg_emb = all_items[neg_items]

		users_emb_ego = self.embedding_user(users)
		pos_emb_ego = self.embedding_item(pos_items)
		neg_emb_ego = self.embedding_item(neg_items)

		return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

	def bpr_loss(self, epoch, users, pos, neg, gum_temp, div_noise, hard):
		
		# Start train gating net
		if epoch > 300:
			self.__gating_freeze(self.gating_net1, True)
			self.__gating_freeze(self.gating_net2, True)   
		
		(users_emb, pos_emb, neg_emb, 
		userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long(), gum_temp, div_noise, hard)
		
		reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
							posEmb0.norm(2).pow(2)  +
							negEmb0.norm(2).pow(2))/float(len(users))
		
		pos_scores = torch.mul(users_emb, pos_emb)
		pos_scores = torch.sum(pos_scores, dim=1)
		neg_scores = torch.mul(users_emb, neg_emb)
		neg_scores = torch.sum(neg_scores, dim=1)
		
		loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
		
		return loss, reg_loss
		
	def forward(self, users, items, gum_temp, div_noise, hard):
		# compute embedding
		all_users, all_items = self.computer(gum_temp, div_noise, hard)

		users_emb = all_users[users]
		items_emb = all_items[items]

		inner_pro = torch.mul(users_emb, items_emb)
		gamma     = torch.sum(inner_pro, dim=1)

		return gamma
