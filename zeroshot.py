import os,random, pickle, json, argparse, time, torch, logging, warnings
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from utils import evaluateQA, evaluateNLI, evaluateNER, evaluatePOS, evaluatePA
from model import BertMetaLearning
from itertools import cycle
from tqdm import tqdm
from datapath import loc

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
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

parser.add_argument('--epochs', type=int, default=2, help='iterations')

parser.add_argument('--seed', type=int, default=0, help='seed for numpy and pytorch')
parser.add_argument('--log_interval', type=int, default=100, help='Print after every log_interval batches')
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--load', type=str, default='', help='')
parser.add_argument('--model_name', type=str, default='model.pt', help='')
parser.add_argument('--grad_clip', type=float, default=1.0)

parser.add_argument('--task',type=str,default='qa_hi')
parser.add_argument('--test', action='store_true')

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

args = parser.parse_args()
print (args)

logger = {'args':vars(args)}
logger['train_loss'] = []
logger['val_loss'] = []
logger['val_metric'] = []
logger['train_metric'] = []

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
	os.makedirs(args.save)

if torch.cuda.is_available():
	if not args.cuda:
		# print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		args.cuda = True

	torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")

def load_data(task_lang):
	[task, lang] = task_lang.split('_')
	if task == 'qa':
		test_corpus = CorpusQA(loc['test'][task_lang][0], args, loc['test'][task_lang][1])
		batch_size = args.qa_batch_size
	elif task == 'sc':
		test_corpus = CorpusSC(loc['test'][task_lang][0], args, loc['test'][task_lang][1])
		batch_size = args.sc_batch_size
	elif task == 'tc':
		test_corpus = CorpusTC(loc['test'][task_lang][0],args)
		batch_size = args.tc_batch_size
	elif task == 'po':
		test_corpus = CorpusPO(loc['test'][task_lang][0],args)
		batch_size = args.po_batch_size
	elif task == 'pa':
		test_corpus = CorpusPA(loc['test'][task_lang][0],args)
		batch_size = args.pa_batch_size

	return test_corpus, batch_size

test_corpus, batch_size = load_data(args.task)
#print(len(train_corpus),len(dev_corpus),len(test_corpus))

# train_dataloader = DataLoader(train_corpus, batch_size = batch_size, pin_memory = True, drop_last = True, shuffle=True)
# dev_dataloader = DataLoader(dev_corpus, batch_size = batch_size, pin_memory = True, drop_last = True)
test_dataloader = DataLoader(test_corpus, batch_size = batch_size, pin_memory = True, drop_last = True)

# print ('Batches | Train %d | Dev %d | Test %d |'%(len(train_dataloader),len(dev_dataloader),len(test_dataloader)))

# steps = args.epochs * len(train_dataloader) + 1

model = BertMetaLearning(args).to(DEVICE)

if args.load != '':
	model = torch.load(args.load)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
	{
		"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		"weight_decay": args.weight_decay,
	},
	{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optim = AdamW(optimizer_grouped_parameters, lr = args.lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup, num_training_steps=10000)

def train(model, task, data):
	to_return = 0.0
	total_loss = 0.0
	t1 = time.time()
	model.train()
	min_task_loss = float('inf')
	for j,batch in enumerate(data):
		optim.zero_grad()
		output = model.forward(task,batch)
		loss = output[0].mean()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		to_return += loss.item()
		total_loss += loss.item()
		optim.step()
		scheduler.step()

		if (j + 1) % args.log_interval == 0:
			print ('batch {:d}/{:d}, time {:6.4f}s, train loss {:10.8f}'.format(j+1,len(data),(time.time() - t1), total_loss/args.log_interval))
			total_loss = 0
			t1 = time.time()

	to_return /= len(data)
	return to_return

def test():
	model.eval()
	if 'qa' in args.task:
		result = evaluateQA(model, test_corpus, 'test_'+args.task, args.save)
		print ('test_f1 {:10.8f}'.format(result['f1']))
		with open(os.path.join(args.save,'test.json'), 'w') as outfile:
			 json.dump(result, outfile)
		test_loss = -result['f1']
	elif 'sc' in args.task:
		test_loss, test_acc = evaluateNLI(model, test_dataloader, DEVICE)
		print ('test_loss {:10.8f} test_acc {:6.4f}'.format(test_loss, test_acc))
	elif 'tc' in args.task:
		test_loss, test_acc = evaluateNER(model, test_dataloader, DEVICE)
		print ('test_loss {:10.8f} test_acc {:6.4f}'.format(test_loss, test_acc))
	elif 'po' in args.task:
		test_loss, test_acc = evaluatePOS(model, test_dataloader, DEVICE)
		print ('test_loss {:10.8f} test_acc {:6.4f}'.format(test_loss, test_acc))
	elif 'pa' in args.task:
		test_loss, test_acc = evaluatePA(model, test_dataloader, DEVICE)
		print ('test_loss {:10.8f} test_acc {:6.4f}'.format(test_loss, test_acc))
	return test_loss

def evaluate(ep, train_loss):
	model.eval()
	if 'qa' in args.task:
		result = evaluateQA(model, dev_corpus, 'val_'+args.task, args.save)
		with open(os.path.join(args.save,'val_' + str(ep) + '.json'), 'w') as outfile:
			json.dump(result, outfile)
		val_f1 = result['f1']
		print ('epoch {:d} val_f1 {:10.8f} train_loss {:10.8f}'.format(ep, val_f1, train_loss))
		logger['val_loss'].append(val_f1)
		val_loss = -val_f1
		val_acc = val_f1
	elif 'sc' in args.task:
		val_loss, val_acc = evaluateNLI(model, dev_dataloader, DEVICE)
		print ('epoch {:d} val_loss {:10.8f} val_acc {:6.4f} train_loss {:10.8f}'.format(ep, val_loss, val_acc, train_loss))
		logger['val_loss'].append(val_loss)
	elif 'pa' in args.task:
		val_loss, val_acc = evaluatePA(model, dev_dataloader, DEVICE)
		print ('epoch {:d} val_loss {:10.8f} val_acc {:6.4f} train_loss {:10.8f}'.format(ep, val_loss, val_acc, train_loss))
		logger['val_loss'].append(val_loss)
	elif 'tc' in args.task:
		val_loss, val_acc = evaluateNER(model, dev_dataloader, DEVICE)
		print ('epoch {:d} val_loss {:10.8f} val_acc {:6.4f} train_loss {:10.8f}'.format(ep, val_loss, val_acc, train_loss))
		logger['val_loss'].append(val_loss)
	elif 'po' in args.task:
		val_loss, val_acc = evaluatePOS(model, dev_dataloader, DEVICE)
		print ('epoch {:d} val_loss {:10.8f} val_acc {:6.4f} train_loss {:10.8f}'.format(ep, val_loss, val_acc, train_loss))
		logger['val_loss'].append(val_loss)
	logger['train_loss'].append(train_loss)
	return val_loss, val_acc

def main():

	try:
		print ("*" * 50)
		print ("Fine Tuning Stage")
		print ("*" * 50)

		min_task_loss = float('inf')
		max_task_acc = 0

		for ep in range(args.epochs):
			model.train()
			train_loss = train(model, args.task, train_dataloader)
			val_loss, val_acc = evaluate(ep, train_loss)

			if 'tc' in args.task or 'sc' in args.task or 'rc' in args.task or 'pa' in args.task:
				logger['val_metric'].append(val_acc)

			if val_loss < min_task_loss:
				print(os.path.join(args.save,args.model_name))
				torch.save(model, os.path.join(args.save,args.model_name))
				min_task_loss = val_loss
				if 'sc' in args.task or 'tc' in args.task or 'rc' in args.task or 'pa' in args.task:
					max_task_acc = val_acc

		with open(os.path.join(args.save,'log.pickle'),'wb') as g:
			pickle.dump(logger,g)

		test()

	except KeyboardInterrupt:

		print ('skipping fine tuning')

		with open(os.path.join(args.save,'log.pickle'),'wb') as g:
			pickle.dump(logger,g)

		test()

if __name__ == '__main__':
	test()
