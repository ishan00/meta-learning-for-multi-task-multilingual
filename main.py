import argparse, time, torch, os, gc, random, pickle, json, copy, logging, warnings, sys
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from utils import evaluateQA, evaluateNLI, evaluateNER, evaluatePA, evaluatePOS
from model import BertMetaLearning
from itertools import cycle
import matplotlib.pyplot as plt
from datapath import loc

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

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--meta_lr', type=float, default=5e-5, help='meta learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--hidden_dims', type=int, default=768, help='')

parser.add_argument('--sc_labels', type=int, default=3, help='')
parser.add_argument('--qa_labels', type=int, default=2, help='')
parser.add_argument('--tc_labels', type=int, default=10, help='')
parser.add_argument('--po_labels', type=int, default=18, help='')
parser.add_argument('--pa_labels', type=int, default=2, help='')

parser.add_argument('--qa_batch_size', type=int, default=8, help='batch size')
parser.add_argument('--sc_batch_size', type=int, default=32, help='batch size')
parser.add_argument('--tc_batch_size', type=int, default=32, help='batch size')
parser.add_argument('--po_batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--pa_batch_size', type=int, default=8, help='batch size')

parser.add_argument('--task_per_queue', type=int, default=8, help='')
parser.add_argument('--update_step', type=int, default=3, help='')
parser.add_argument('--beta', type=float, default=1.0, help='')
parser.add_argument('--meta_epochs', type=int, default=3, help='iterations')

parser.add_argument('--seed', type=int, default=42, help='seed for numpy and pytorch')
parser.add_argument('--log_interval', type=int, default=200, help='Print after every log_interval batches')
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--load', type=str, default='', help='')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--meta_tasks', type=str, default='sc,pa,qa,tc,po')

parser.add_argument('--sampler',type=str,default='uniform_batch',choices=['uniform_batch'])
parser.add_argument('--temp', type=float, default=1.0)

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

for tt in task_types:
	if '_' in tt:
		list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
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
	elif 'po' in k:
		train_corpus[k] = CorpusPO(loc['train'][k][0], args)
		dev_corpus[k] = CorpusPO(loc['dev'][k][0], args)
	elif 'pa' in k:
		train_corpus[k] = CorpusPA(loc['train'][k][0], args)
		dev_corpus[k] = CorpusPA(loc['dev'][k][0], args)

train_dataloaders = {}
dev_dataloaders = {}

for k in list_of_tasks:
	batch_size = args.qa_batch_size if 'qa' in k else \
				args.sc_batch_size if 'sc' in k else \
				args.tc_batch_size if 'tc' in k else \
				args.po_batch_size if 'po' in k else \
				args.pa_batch_size
	train_dataloaders[k] = DataLoader(train_corpus[k], batch_size = batch_size, shuffle = True, pin_memory = True)
	dev_dataloaders[k] = DataLoader(dev_corpus[k], batch_size = batch_size, pin_memory = True)

model = BertMetaLearning(args).to(DEVICE)

if args.load != '':
	model = torch.load(args.load)

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
	}
]

optim = AdamW(optimizer_grouped_parameters, lr = args.meta_lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optim, num_warmup_steps=args.warmup, num_training_steps=steps
)

logger = {}
logger['total_val_loss'] = []
logger['val_loss'] = {k:[] for k in list_of_tasks}
logger['train_loss'] = []
logger['args'] = args

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

sampler = UniformBatchSampler()

print (sampler.init_p)

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

	min_task_losses =  {
		"qa" : float('inf'),
		"sc" : float('inf'),
		"po" : float('inf'),
		"tc" : float('inf'),
		"pa" : float('inf'),
	}

	global_time = time.time()

	try:

		for j,metabatch in enumerate(sampler):
			if j > steps: break
			loss, _ = metastep(model, metabatch)
			total_loss += sum([y for (x,y) in loss])/len(loss)
			logger['train_loss'].append(sum([y for (x,y) in loss])/len(loss))

			if (j + 1) % args.log_interval == 0:

				val_loss_dict, val_loss_total = evaluateMeta(model)

				logger['total_val_loss'].append(val_loss_total)
				for task in val_loss_dict.keys():
					logger['val_loss'][task].append(val_loss_dict[task])

				total_loss /= args.log_interval
				print('Val Loss Dict : ',val_loss_dict)

				loss_per_task = {}
				for task in val_loss_dict.keys():
					if task[:2] in loss_per_task.keys():
						loss_per_task[task[:2]] = loss_per_task[task[:2]] + val_loss_dict[task]
					else:
						loss_per_task[task[:2]] = val_loss_dict[task]

				print('Time : %f , Step  : %d , Train Loss : %f, Val Loss : %f' % (time.time() - global_time,j+1,total_loss,val_loss_total))
				print('===============================================')
				global_time = time.time()

				for task in loss_per_task.keys():
					if loss_per_task[task] < min_task_losses[task]:
						torch.save(model,os.path.join(args.save,"model_"+task+".pt"))
						min_task_losses[task] = loss_per_task[task]
						print("Saving "+task+"  Model")
				total_loss = 0

				with open(os.path.join(args.save,'logger.pickle'),'wb') as f:
					pickle.dump(logger,f)
			scheduler.step()

	except KeyboardInterrupt:
		print ('skipping meta learning')

	with open(os.path.join(args.save,'logger.pickle'),'wb') as f:
		pickle.dump(logger,f)

if __name__ == '__main__':
	main()

