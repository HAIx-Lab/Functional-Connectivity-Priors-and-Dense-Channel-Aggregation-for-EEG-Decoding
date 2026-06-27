import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from model.TmpEncoder_stage5_clean import *
import os
import mne

from datasets.faced_dataset import LoadDataset as faced
from datasets.bciciv2a_dataset import LoadDataset as bciciv2a
from datasets.mumtaz_dataset import LoadDataset as mumtaz
from datasets.tuab_dataset import LoadDataset as tuab
from datasets.tuev_dataset import LoadDataset as tuev
from datasets.siena_dataset import LoadDataset as siena
from datasets.mentalmath_dataset import LoadDataset as math
from datasets.speech_dataset import LoadDataset as speech

from model.TmpEncoder_stage5_clean import Final
from model.TmpEncoder_continuous import Final as Final_continuous
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
		precision_recall_curve, auc, r2_score, mean_squared_error

import ast


hcp_positions_path = os.path.expanduser('~/simpletmp/data/HCP/positions_100_7.txt')
connectivity_path = os.path.expanduser('~/simpletmp/processed_data/connectivity_matrix.txt')

class Evaluator:
	def __init__(self, params, data_loader, model, epoch):
		self.params = params
		self.data_loader = data_loader
		self.device = torch.device(f'cuda:{self.params.cuda}') if torch.cuda.is_available() else 'cpu'
		self.model = model.to(self.device)

		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)		
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)

	def get_metrics_for_multiclass(self):
		self.model.eval()

		truths = []
		preds = []
		losses = []
		
		with torch.no_grad():
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Multiclass")):
				x = x.to(self.device)
				y = y.to(self.device).long() 

				pred = self.model(x)

				pred_y = torch.max(pred, dim=-1)[1]

				loss = self.criterion_for_multiclass(pred, y)
				losses.append(loss.item())

				truths.extend(y.cpu().view(-1).numpy())
				preds.extend(pred_y.cpu().view(-1).numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		acc = balanced_accuracy_score(truths, preds)
		f1 = f1_score(truths, preds, average='weighted')
		kappa = cohen_kappa_score(truths, preds)

		return acc, kappa, f1, mean_loss

	def get_metrics_for_binaryclass(self):
		self.model.eval() 

		truths = []
		preds = []
		scores = []
		losses = []

		with torch.no_grad():
			for i, (x, y) in enumerate(tqdm(self.data_loader, mininterval=1, desc="Evaluating Binary")):
				x = x.to(self.device)
				y = y.to(self.device).float()

				logit = self.model(x)

				logit_flat = logit.view(-1)
				y_flat = y.view(-1)

				score_y = torch.sigmoid(logit_flat)
				pred_y = torch.gt(score_y, 0.5).long()

				loss = self.criterion_for_binary_class(logit_flat, y_flat)
				losses.append(loss.item())

				truths.extend(y_flat.cpu().long().numpy())
				preds.extend(pred_y.cpu().numpy())
				scores.extend(score_y.cpu().numpy())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)
		scores = np.array(scores)

		acc = balanced_accuracy_score(truths, preds)
		cohen = cohen_kappa_score(truths, preds)

		precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
		pr_auc = auc(recall, precision)

		try:
				roc_auc = roc_auc_score(truths, scores)
		except ValueError:
				roc_auc = 0.0 

		return acc, pr_auc, roc_auc, cohen, mean_loss

	def get_metrics_for_regression(self):
		self.model.eval() 

		truths = []
		preds = []
		losses = []

		with torch.no_grad(): 
			for x, y in tqdm(self.data_loader, mininterval=1):
				x = x.to(self.device)
				y = y.to(self.device).float()

				pred = self.model(x)

				truths += y.cpu().squeeze(-1).numpy().tolist()
				preds += pred.cpu().squeeze(-1).numpy().tolist()

				loss = self.criterion_for_regression(pred.squeeze(-1), y.squeeze(-1))
				losses.append(loss.item())

		mean_loss = np.mean(losses)
		truths = np.array(truths)
		preds = np.array(preds)

		try:
			corrcoef = np.corrcoef(truths, preds)[0, 1]
		except Exception:
			corrcoef = 0.0

		r2 = r2_score(truths, preds)
		rmse = mean_squared_error(truths, preds) ** 0.5

		return corrcoef, r2, rmse, mean_loss

class ConfiguredModel(nn.Module):
	def __init__(self, model, params):
		super().__init__()

		self.backbone = model
		self.params = params

		self.FFN = nn.Sequential(
								nn.Linear(200, self.params.nc),	
						)

	def forward(self, x):
			Bz, num_chans, num_patches, patch_size = x.shape

			emb = self.backbone(x)
			emb = emb.mean(dim=(1, 2))

			out = self.FFN(emb)
			out = out.reshape(Bz, self.params.nc)

			return out

class FineTune_Trainer(object):
	def __init__(self, params, data_loader, model):
		super().__init__()

		self.params = params
		self.model = model

		self.device = torch.device(f'cuda:{self.params.cuda}' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)

		backbone_parameters = []
		other_parameters = []
		
		self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

		self.data_loader = data_loader
		self.length = len(data_loader['train'])

		self.criterion_for_binary_class = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
		self.criterion_for_multiclass = nn.CrossEntropyLoss().to(self.device)
		self.criterion_for_regression = nn.MSELoss().to(self.device)

		warmup_steps = (self.params.epochs * 0.4) // 10 * self.length
		total_steps = self.params.epochs * self.length
		main_steps = total_steps - warmup_steps

		self.optimizer_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=warmup_steps, start_factor=0.1, end_factor=1)
		self.optimizer_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = main_steps, eta_min=1e-7)

		self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.optimizer_scheduler_warmup, self.optimizer_scheduler_main], milestones=[warmup_steps])

	def train_for_binaryclass(self):

		best_loss = float('inf')

		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')

			losses = []
			self.model.train()

			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):

				self.optimizer.zero_grad()

				x = x.to(self.device)
				label = label.to(self.device)

				logit = self.model(x)
				logit = logit.squeeze(-1)
				label = label.squeeze(-1)
				loss = self.criterion_for_binary_class(logit, label.float())

				loss.backward()

				if self.params.clip_value > 0:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

				self.optimizer.step()
				self.scheduler.step()

				losses.append(loss.item())

			mean_loss = np.mean(losses)

			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}_{self.params.dataset}_{self.params.seed}_{self.params.config}.pth')
			torch.save(self.model.state_dict(), save_path)

			print('model saved @ ', save_path)

			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')

			with torch.no_grad():
				self.model.eval()

				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				b_acc, pr_auc, auroc, cohen, mean_loss = evaluator.get_metrics_for_binaryclass()

				print(f'b_acc: {b_acc}, pr_auc: {pr_auc}, auroc: {auroc}, cohen:{cohen}, mean_loss: {mean_loss}')

	def train_for_multiclass(self):
		best_loss = float('inf')

		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')

			losses = []
			self.model.train()

			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):

				self.optimizer.zero_grad()

				x = x.to(self.device)
				label = label.to(self.device).long()

				logit = self.model(x)   # logit is(Bz, self.num_classes)
				loss = self.criterion_for_multiclass(logit, label)

				loss.backward()

				if self.params.clip_value > 0:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

				self.optimizer.step()
				self.scheduler.step()

				losses.append(loss.item())
				print(np.argmax(logit.detach().cpu().numpy(), axis=1))

			mean_loss = np.mean(losses)

			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}_{self.params.seed}_{self.params.mode}_{self.params.dataset}_{self.params.config}.pth')
			torch.save(self.model.state_dict(), save_path)

			print('model saved @ ', save_path)

			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')

			with torch.no_grad():
				self.model.eval()

				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				b_acc, cohen, f1, mean_loss = evaluator.get_metrics_for_multiclass()

				print(f'b_acc: {b_acc}, cohen: {cohen}, f1: {f1}, mean_loss: {mean_loss}')

	def train_for_regression(self):
		best_loss = float('inf')

		for epoch in range(self.params.epochs):
			print(f'Epoch {epoch} starts')

			losses = []
			self.model.train()

			for i, (x, label) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
				self.optimizer.zero_grad()

				x = x.to(self.device)
				label = label.to(self.device).float()

				logit = self.model(x)   

				logit = logit.squeeze(-1)
				label = label.squeeze(-1)

				loss = self.criterion_for_regression(logit, label)

				loss.backward()

				if self.params.clip_value > 0:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

				self.optimizer.step()
				self.scheduler.step()

				losses.append(loss.item())

			mean_loss = np.mean(losses)

			save_path = os.path.join(self.params.model_dir, f'finetune_weights_{epoch}_{self.params.seed}_{self.params.mode}_{self.params.dataset}.pth')
			torch.save(self.model.state_dict(), save_path)

			print('model saved @ ', save_path)

			print(f'Epoch {epoch} model trained. Now commencing testing.')
			print(f'loss: {mean_loss}')

			with torch.no_grad():
				self.model.eval()

				evaluator = Evaluator(params=self.params, data_loader=self.data_loader['val'], model=self.model, epoch=epoch)
				corrcoef, r2, rmse, mean_loss = evaluator.get_metrics_for_regression()

				print(f'corrcoef: {corrcoef}, r2: {r2}, rmse: {rmse}, mean_loss: {mean_loss}')

def setup_seed(seed):
		torch.manual_seed(seed) 
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

def sorted_maps(ch_names):
	montage = mne.channels.make_standard_montage('standard_1005')
	all_pos = montage.get_positions()['ch_pos']
  
	pos_array = np.array([all_pos[ch] for ch in ch_names]) * 1000
	
	brain_regions = np.loadtxt(hcp_positions_path)
	used_pos = []
	
	for ch in ch_names:
		pos = all_pos[ch] * 1000
		idx = np.argmin(np.sum((brain_regions - pos)**2, axis=1))
		
		used_pos.append(idx)
	 
	correlations = np.loadtxt(connectivity_path)	
	
	sub_corr = correlations[np.ix_(used_pos, used_pos)]	
	sorted_map = np.argsort(-sub_corr, axis=1)
	
	return torch.tensor(sorted_map)
	
	
def main():
	# It is ok to not have a masking parameter here. Because it is not like the model inherently masks. It is the pretraining script that is masking. Here, we don't have that if block, so nothing to worry about.
	parser = argparse.ArgumentParser(description='Finetuning EEG FM')
	
	parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
	parser.add_argument('--parallel', action='store_false', default=True, help='want gpu?')
	
	parser.add_argument('--multi_lr', action='store_false', default=True, help='variable learning rates for big model and small layer')
	parser.add_argument('--lr', type=float, default=5e-6, help='lr')
	
	parser.add_argument('--weight_decay', type=float, default=5e-2, help='wt dikey')
	parser.add_argument('--epochs', type=int, default=50, help='num_epochs')
	
	parser.add_argument('--clip_value', type=float, default=1, help='clip value')
	parser.add_argument('--model_dir', type=str, default='~/simpletmp/saved_fm', help='weight storage')
	
	parser.add_argument('--dataset_dir', type=str, default='~/CantusCerebra/data/TUAB/edf/process_refine', help='your 3 json files folder')
	parser.add_argument('--seed', type=int, default=42, help='seed')
	
	parser.add_argument('--cuda', type=int, default=0, help='What is the primary gpu?')
	parser.add_argument('--avail_gpus', type=str, default='0', help='Provide the explicit numbers a/c the motherboard of available GPUs.')

	parser.add_argument('--batch_size', type=int, default=64, help='Bz')
	
	parser.add_argument('--in_dim', type=int, default=200, help='Number of samples in 1s raw')		
	parser.add_argument('--out_dim', type=int, default=200, help='Output dimension')
	
	parser.add_argument('--d_model', type=int, default=200, help='Model operating dimension')
	parser.add_argument('--d_ffn', type=int, default=800, help='Standard 2-layer FFN dimensions')
	
	parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
	parser.add_argument('--num_heads', type=int, default=8, help='Number of Heads in MHSA')
	
	parser.add_argument('--convolution_set', type=str, default="[(1,), (3,), (5,)]", help='Concentrated convolution sizes. < num_chans')
	parser.add_argument('--seq_len', type=int, default=30, help='num_patches')
	
	parser.add_argument('--is_causal', action='store_true', help='If you want causal Temporal Attention')
	parser.add_argument('--need_key_padding', action='store_true', help='if any padding that could be added is to be ignored')
	
	parser.add_argument('--stride', type=int, default=1, help='stride for temp convs')

	parser.add_argument('--mode', type=str, default='bin, mul, reg')
	parser.add_argument('--nc', type=int, default=1, help='num_classes in the dataset')
	
	parser.add_argument('--dataset', default='FACED', help='FACED, BCICIV2a, Mumtaz2016, TUEV, TUAB, siena')
	parser.add_argument('--config', default='random', help='random,cont,hcp')
	
	params = parser.parse_args()
	setup_seed(params.seed)	
	
	params.model_dir = os.path.expanduser(params.model_dir)
	params.dataset_dir = os.path.expanduser(params.dataset_dir)
	
	if params.dataset == 'FACED':
		dataset = faced(params)
		sorted_map = sorted_maps([
			"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", 
			"Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", 
			"Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"
		])
		params.nc = 9
		random_map = torch.randint_like(sorted_map, high=32)
	if params.dataset == 'BCICIV2a':
		dataset = bciciv2a(params)
		sorted_map = sorted_maps([
			"Fz", 
			"FC3", "FC1", "FCz", "FC2", "FC4", 
			"C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
			"CP3", "CP1", "CPz", "CP2", "CP4", 
			"P1", "Pz", "P2", "POz"
		])
		params.nc = 5
		random_map = torch.randint_like(sorted_map, high=22)
	if params.dataset == 'Mumtaz2016':
		dataset = mumtaz(params)
		sorted_map = sorted_maps([
			'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
			'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
		])
		random_map = torch.randint_like(sorted_map, high=19)
		params.nc = 1
	if params.dataset == 'TUEV':
		dataset = tuev(params)
		sorted_map = sorted_maps([
			'Fp1', 'F7', 'T7', 'P7', 'O1', 
			'Fp2', 'F8', 'T8', 'P8', 'O2', 
			'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
		])
		params.nc = 6
		random_map = torch.randint_like(sorted_map, high=16)
	if params.dataset == 'TUAB':
		dataset = tuab(params)
		sorted_map = sorted_maps([
			'Fp1', 'F7', 'T7', 'P7', 'O1', 
			'Fp2', 'F8', 'T8', 'P8', 'O2', 
			'F3', 'C3', 'P3', 'F4', 'C4', 'P4'
		])
		params.nc = 1
		random_map = torch.randint_like(sorted_map, high=16)
	if params.dataset == 'siena':
		dataset = siena(params)
		sorted_map = sorted_maps([
	'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 'P7', 
	'FC1', 'FC5', 'CP1', 'CP5', 'F9', 'Fz', 'Cz', 'Pz', 
	'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 	
	'FC2', 'FC6', 'CP2', 'CP6', 'F10'])
		params.nc = 1
		random_map = torch.randint_like(sorted_map, high=29)
	if params.dataset == 'MentalArithmetic':
		dataset = math(params)
		sorted_map = sorted_maps([
		 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
		 'T7', 'T8', 'C3', 'C4', 'P7', 'P8',
		 'P3', 'P4', 'O1', 'O2',
		 'Fz', 'Cz', 'Pz'
		])
		params.nc = 1
		random_map = torch.randint_like(sorted_map, high=19)
	if params.dataset == 'speech':
		dataset = speech(params)
		sorted_map = sorted_maps([
			'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
			'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 
			'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 
			'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 
			'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 
			'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 
			'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 
			'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'
		])
		params.nc = 5
		random_map = torch.randint_like(sorted_map, high=64)
	
	data_loader = dataset.get_data_loader()
	if(params.config == 'random'):
		mp = random_map
	else:
		mp = sorted_map
	
	if(params.config == 'cont'):
		model=Final_continuous(mp, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, num_heads=params.num_heads)
	else:
		model=Final(mp, in_dim=params.in_dim, out_dim=params.out_dim, d_model=params.d_model, num_layers=params.num_layers, convolution_set=ast.literal_eval(params.convolution_set), stride=params.stride, dropout=params.dropout, d_ffn=params.d_ffn, num_heads=params.num_heads)
		
	model = ConfiguredModel(model, params=params)
	
	trainer = FineTune_Trainer(params=params, model=model, data_loader=data_loader)	
	
	if params.mode == 'bin':
		trainer.train_for_binaryclass()
	if params.mode == 'mul':
		trainer.train_for_multiclass()
	if params.mode == 'reg':
		trainer.train_for_regression()
	
if __name__== '__main__':
	main()
