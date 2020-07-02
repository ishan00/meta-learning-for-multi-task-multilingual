import torch
import time
import argparse
import numpy as np
import logging
from torch.utils.data import DataLoader, RandomSampler
from data import CorpusQA, CorpusSC, CorpusTC, CorpusRC, CorpusPA
from utils import evaluateQA, evaluateNLI, evaluateNER, evaluatePA
from model import BertMetaLearning
import os
import gc
import random
from itertools import cycle
import matplotlib.pyplot as plt
import pickle,json
import copy
import warnings
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from datapath import loc
import sys

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

parser = argparse.ArgumentParser()

parser.add_argument('--meta_lr', type=float, default=5e-5, help='meta learning rate')
parser.add_argument('--sigma', action='store_true',help='use sigma weighing of loss')
parser.add_argument('--sigma_lr',type=float,default=1e-3, help='sigma learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--hidden_dims', type=int, default=768, help='')

parser.add_argument('--sc_labels', type=int, default=3, help='')
parser.add_argument('--qa_labels', type=int, default=2, help='')
parser.add_argument('--tc_labels', type=int, default=10, help='')
parser.add_argument('--rc_labels', type=int, default=1, help='')
parser.add_argument('--pa_labels', type=int, default=2, help='')

parser.add_argument('--qa_batch_size', type=int, default=8, help='batch size')
parser.add_argument('--sc_batch_size', type=int, default=32, help='batch size')
parser.add_argument('--tc_batch_size', type=int, default=32, help='batch size')
parser.add_argument('--rc_batch_size', type=int, default=8, help='batch size')
parser.add_argument('--pa_batch_size', type=int, default=32, help='batch size')

parser.add_argument('--task_per_queue', type=int, default=8, help='')
parser.add_argument('--update_step', type=int, default=3, help='')
parser.add_argument('--beta', type=float, default=1.0, help='')
parser.add_argument('--meta_epochs', type=int, default=5, help='iterations')

parser.add_argument('--seed', type=int, default=42, help='seed for numpy and pytorch')
parser.add_argument('--log_interval', type=int, default=200, help='Print after every log_interval batches')
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--load', type=str, default='', help='')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--meta_tasks', type=str, default='sc,pa,qa,tc,rc')

parser.add_argument('--sampler',type=str,default='uniform_batch',choices=['uniform_batch'])
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--update_dds', type=int,default=10)
parser.add_argument('--dds_lr', type=float,default=0.01)
parser.add_argument('--target_task',type=str,default='qa_hi')

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

args = parser.parse_args()

if not os.path.exists(args.save):
	os.makedirs(args.save)
	
class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open(os.path.join(args.save,"output.txt"), "w")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.log.flush()
		
sys.stdout = Logger()
print (args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
	if not args.cuda:
		args.cuda = True

	torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")

task_types = args.meta_tasks.split(',')
list_of_tasks = []

for tt in loc['train'].keys():
	if tt[:2] in task_types:
		list_of_tasks.append(tt)

num_tasks = len(list_of_tasks)
print (list_of_tasks)

train_corpus = {}
dev_corpus = {}

for k in list_of_tasks:
	if 'qa' in k:
		train_corpus[k] = CorpusQA(loc['train'][k][0], args, loc['train'][k][1])
		dev_corpus[k] = CorpusQA(loc['dev'][k][0], args, loc['dev'][k][1])
	elif 'sc' in k:
		train_corpus[k] = CorpusSC(loc['train'][k][0], args, loc['train'][k][1])
		dev_corpus[k] = CorpusSC(loc['dev'][k][0], args, loc['dev'][k][1])
	elif 'tc' in k:
		train_corpus[k] = CorpusTC(loc['train'][k][0], args)
		dev_corpus[k] = CorpusTC(loc['dev'][k][0], args)
	elif 'rc' in k:
		train_corpus[k] = CorpusRC(loc['train'][k][0], args)
		dev_corpus[k] = CorpusRC(loc['dev'][k][0], args)
	elif 'pa' in k:
		train_corpus[k] = CorpusPA(loc['train'][k][0], args)
		dev_corpus[k] = CorpusPA(loc['dev'][k][0], args)

train_dataloaders = {}
dev_dataloaders = {}
psi_train_dataloaders = {}

for k in list_of_tasks:
	batch_size = args.qa_batch_size if 'qa' in k else \
				args.sc_batch_size if 'sc' in k else \
				args.tc_batch_size if 'tc' in k else \
				args.rc_batch_size if 'rc' in k else \
				args.pa_batch_size
	train_dataloaders[k] = DataLoader(train_corpus[k], batch_size = batch_size, shuffle = True, pin_memory = True)
	dev_dataloaders[k] = DataLoader(dev_corpus[k], batch_size = batch_size, pin_memory = True)
	psi_train_dataloaders[k] = DataLoader(train_corpus[k], batch_size = batch_size, pin_memory = True, sampler = RandomSampler(train_corpus[k]))

psi_dev_dataloader = DataLoader(dev_corpus[args.target_task], batch_size = batch_size, pin_memory = True, sampler = RandomSampler(dev_corpus[args.target_task]))

list_of_psi_train_iters = {k:iter(psi_train_dataloaders[k]) for k in list_of_tasks}
psi_dev_iters = iter(psi_dev_dataloader)

psis = {k:torch.log(torch.tensor(len(train_dataloaders[k])*1.0,device = DEVICE)) for k in list_of_tasks}

print ('initial',psis)

model = BertMetaLearning(args)
model = model.to(DEVICE)

# if args.load != '':
# 	model = torch.load(args.load)

params = list(model.parameters())

steps = args.meta_epochs * sum([len(train_dataloaders[x]) for x in list_of_tasks]) // (args.task_per_queue * args.update_step)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
	{
		"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		"weight_decay": args.weight_decay,
		"lr": args.meta_lr
	},
	{
		"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
		"weight_decay": 0.0,
		"lr": args.meta_lr,
	},
]

optim = AdamW(optimizer_grouped_parameters, lr = args.meta_lr, eps=args.adam_epsilon)
# optim1 = AdamW(optimizer_grouped_parameters, lr = args.meta_lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optim, num_warmup_steps=args.warmup, num_training_steps=steps
)

logger = {}
logger['total_val_loss'] = []
logger['val_loss'] = {k:[] for k in list_of_tasks}
logger['psis'] = []
logger['train_loss'] = []
logger['args'] = args
logger['saved_at'] = None

def get_batch(dataloader_iter, dataloader):
	try:
		batch = next(dataloader_iter)
	except StopIteration:
		dataloader_iter = iter(dataloader)
		batch = next(dataloader_iter)
	return batch

class Sampler:
	def __init__(self, p, steps, func):
		# Sampling Weights
		self.init_p = p
		self.total_steps = steps
		self.func = func
		self.curr_step = 0

		self.update_step = args.update_step
		self.task_per_queue = args.task_per_queue
		self.list_of_tasks = list_of_tasks
		self.list_of_iters = {k:iter(train_dataloaders[k]) for k in self.list_of_tasks}

	def __iter__(self):
		return self

	def __next__(self):
		curr_p = self.func(self.init_p, self.list_of_tasks, self.curr_step, self.total_steps)
		self.curr_step += 1
		tasks = np.random.choice(self.list_of_tasks,self.task_per_queue,p=curr_p)
		queue = []
		for i in range(self.task_per_queue):
			l = {'task':tasks[i],'data':[]}
			for _ in range(self.update_step):
				l['data'].append(get_batch(self.list_of_iters[tasks[i]], train_dataloaders[tasks[i]]))
			queue.append(l)
		return queue

def identity(x,y,z,w): return x

def UniformBatchSampler():
	p = np.array([len(train_dataloaders[y])*1.0/sum([len(train_dataloaders[x]) for x in list_of_tasks]) for y in list_of_tasks])
	p_temp = np.power(p, 1.0/args.temp)
	p_temp = p_temp / np.sum(p_temp)
	sampler = iter(Sampler(p_temp, steps, identity))
	return sampler

def MultiDDSSampler():
	p = np.array([len(train_dataloaders[y])*1.0/sum([len(train_dataloaders[x]) for x in list_of_tasks]) for y in list_of_tasks])
	return iter(Sampler(p,steps,multiDDSUpdate))

def multiDDSUpdate(init_p, list_of_tasks, curr_step, total_steps):

	p = np.array([torch.exp(psis[x]).item() for x in list_of_tasks])
	p = p/np.sum(p)
	
	if (curr_step+1)%args.update_dds == 0:
		old_vars_dds = []
		for param in model.parameters():
			old_vars_dds.append(param.data.clone())
		rewards = {}
		mean_reward = 0
		for task in list_of_tasks:
			reward = 0
			train_batch = get_batch(list_of_psi_train_iters[task], psi_train_dataloaders[task])
			optim.zero_grad()
			output = model.forward(task, train_batch)
			loss = output[0].mean()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			g_train = []
			for idx, param in enumerate(model.parameters()):
				if param.grad is not None:
					g_train.append(param.grad.clone())
				else:
					g_train.append(torch.zeros_like(param))
			
			sq_train = torch.tensor(0.0,device=DEVICE)
			for idx in range(len(g_train)):
				sq_train += torch.sum(g_train[idx] * g_train[idx])
			optim.step()
			optim.zero_grad()
			dev_batch = get_batch(psi_dev_iters, psi_dev_dataloader)
			output = model.forward(args.target_task, dev_batch)
			loss = output[0].mean()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			dot_pdt = torch.tensor(0.0,device=DEVICE)
			sq_dev = torch.tensor(0.0,device=DEVICE)
			
			for idx, param in enumerate(model.parameters()):
				if param.grad is not None:
					g_dev = param.grad.clone()
				else:
					g_dev = torch.zeros_like(param)
				dot_pdt += torch.sum(g_dev * g_train[idx])
				sq_dev += torch.sum(g_dev * g_dev)

			print (task, dot_pdt, sq_dev, sq_train)

			reward = dot_pdt / torch.sqrt(sq_dev * sq_train)
			rewards[task] = reward
			mean_reward += reward/len(list_of_tasks)
			for idx,param in enumerate(model.parameters()):
				param.data = old_vars_dds[idx].data.clone()

		for idx1, task1 in enumerate(list_of_tasks):
			for idx2, task2 in enumerate(list_of_tasks):
				if task1 == task2:
					psis[task2] += args.dds_lr * (1-p[idx2]) * rewards[task1]
				else:
					psis[task2] -= args.dds_lr * p[idx2] * rewards[task1]
		
		p = np.array([torch.exp(psis[x]).item() for x in list_of_tasks])
		p = p/np.sum(p)
		print(['{:s},{:6.4f},{:6.4f},{:6.4f}'.format(task,p[idx],rewards[task].item(),psis[task].item()) for idx,task in enumerate(list_of_tasks)])
		logger['psis'].append(['{:s},{:6.4f},{:6.4f},{:6.4f}'.format(task,p[idx],rewards[task].item(),psis[task].item()) for idx,task in enumerate(list_of_tasks)])
		print('======================================================================')

	return p

sampler = MultiDDSSampler()
print(['{:s},{:6.4f}'.format(task,psis[task].item()) for idx,task in enumerate(list_of_tasks)])

def metastep(model, queue):
	t1 = time.time()
	n = len(queue)
	old_vars = []
	running_vars = []
	for param in model.parameters():
		old_vars.append(param.data.clone())
	losses = [[0 for _ in range(args.update_step)] for _ in range(n)]
	for i in range(n):
		for k in range(args.update_step):
			t = time.time()
			optim.zero_grad()
			output = model.forward(queue[i]['task'],queue[i]['data'][k])
			loss = output[0].mean()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			optim.step()

			losses[i][k] += loss.item()

		if running_vars == []:
			for _,param in enumerate(model.parameters()):
				running_vars.append(param.data.clone())
		else:
			for idx,param in enumerate(model.parameters()):
				running_vars[idx].data += param.data.clone()

		for idx,param in enumerate(model.parameters()):
			param.data = old_vars[idx].data.clone()

	for param in running_vars:
		param /= n

	for idx,param in enumerate(model.parameters()):
		param.data = old_vars[idx].data + args.beta * (running_vars[idx].data - old_vars[idx].data)
	return [(queue[i]['task'],sum(l)/len(l)) for i,l in enumerate(losses)], time.time() - t1

def evaluate(model, task, data):
# 	model.eval()
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		for j,batch in enumerate(data):
			output = model.forward(task,batch)
			loss = output[0].mean()
			total_loss += loss.item()
		total_loss /= len(data)
		return total_loss

def evaluateMeta(model):
	loss_dict = {}
	total_loss = 0
	model.eval()
	for task in list_of_tasks:
		loss = evaluate(model, task, dev_dataloaders[task])
		loss_dict[task] = loss
		total_loss += loss
	return loss_dict, total_loss

def main():

	# Meta learning stage

	print ("*" * 50)
	print ("Meta Learning Stage")
	print ("*" * 50)

	print ('Training for %d metasteps'%steps)


	total_loss = 0
	
	min_val_loss = float('inf')

	global_time = time.time()

	try:

		for j,metabatch in enumerate(sampler):
			if j > steps: break
			loss, _ = metastep(model, metabatch)
			total_loss += sum([y for (x,y) in loss])/len(loss)
			logger['train_loss'].append(sum([y for (x,y) in loss])/len(loss))

			if (j + 1) % args.log_interval == 0:
				val_loss = evaluate(model, args.target_task, dev_dataloaders[args.target_task])
				logger['val_loss'][args.target_task].append(val_loss)

# 				logger['total_val_loss'].append(val_loss_total)
# 				for task in val_loss_dict.keys():
# 					logger['val_loss'][task].append(val_loss_dict[task])

# 				total_loss /= args.log_interval
# 				print('Val Loss Dict : ',val_loss_dict)

# 				for task in val_loss_dict.keys():
# 					if 'qa' in task:
# 						result = evaluateQA(model, dev_corpus[task], 'val_'+task, args.save)
# 						with open(os.path.join(args.save,'val_' + str(j + 1) + '_'  + task + '.json'), 'w') as outfile:
# 							json.dump(result, outfile)
# 						val_f1 = result['f1']
# 						print ('val_f1 {:10.8f}'.format(val_f1))
# 						logger['val_f1'].append(val_f1)
# 						val_loss_dict[task] = -val_f1

				print('Time : %f , Step  : %d , Train Loss : %f, Val Loss : %f' % (time.time() - global_time,j+1,total_loss,val_loss))
				print('===============================================')
				global_time = time.time()

				if val_loss < min_val_loss:
					torch.save(model,os.path.join(args.save,"model_"+args.target_task+".pt"))
					min_val_loss = val_loss
					print("Saving "+args.target_task+"  Model")
					logger['saved_at'] = j + 1
				
				with open(os.path.join(args.save,'logger.pickle'),'wb') as f:
					pickle.dump(logger,f)
				
				total_loss = 0
			scheduler.step()

	except KeyboardInterrupt:
		print ('skipping meta learning')

	with open(os.path.join(args.save,'logger.pickle'),'wb') as f:
		pickle.dump(logger,f)

if __name__ == '__main__':
	main()




