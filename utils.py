import os
import torch
from squad_metrics import (
		compute_predictions_log_probs,
		compute_predictions_logits,
		squad_evaluate,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mentor(nn.Module):
    def __init__(self, nb_classes):
        super(Mentor, self).__init__()

        self.label_emb = nn.Embedding(nb_classes, 2)
        self.percent_emb = nn.Embedding(100, 5)

        self.lstm = nn.LSTM(2, 10, bidirectional=True)
        self.fc1 = nn.Linear(27, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, data):
        label, pt, l = data
        x_label = self.label_emb(label)
        x_percent = self.percent_emb(pt)
        h0 = torch.rand(2, label.size(0), 10).to(DEVICE)
        c0 = torch.rand(2, label.size(0), 10).to(DEVICE)

        output, (hn,cn) = self.lstm(l, (h0,c0))
        output = output.sum(0).squeeze()
        x = torch.cat((x_label, x_percent, output), dim=1)
        z = torch.tanh(self.fc1(x))
        z = torch.sigmoid(self.fc2(z))
        return z

class SquadResult(object):
	def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
		self.start_logits = start_logits
		self.end_logits = end_logits
		self.unique_id = unique_id

		if start_top_index:
			self.start_top_index = start_top_index
			self.end_top_index = end_top_index
			self.cls_logits = cls_logits

def to_list(tensor):
	return tensor.detach().cpu().tolist()

def evaluateQA(model, corpus, task, path):
	dataset, examples, features = corpus.dataset, corpus.examples, corpus.features
	tokenizer = corpus.tokenizer
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(dataset)
	eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=12)

	all_results = []

	for batch in eval_dataloader:
		model.eval()

		with torch.no_grad():
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"token_type_ids": batch[2],
				"answer_start": None,
				"answer_end": None,
			}

			example_indices = batch[3]

			outputs = model('qa',inputs)

			for i, example_index in enumerate(example_indices):
				eval_feature = features[example_index.item()]
				unique_id = int(eval_feature.unique_id)

				output = [to_list(output[i]) for output in outputs]

				start_logits, end_logits = output
				result = SquadResult(unique_id, start_logits, end_logits)

				all_results.append(result)

		# Compute predictions
	output_prediction_file = os.path.join(path,"predictions_{}.json".format(task))
	output_nbest_file = os.path.join(path,"nbest_predictions_{}.json".format(task))

	output_null_log_odds_file = None

	predictions = compute_predictions_logits(
		examples,
		features,
		all_results,
		20,
		30,
		True,
		output_prediction_file,
		output_nbest_file,
		output_null_log_odds_file,
		False,
		False,
		0.0,
		tokenizer,
	)

	# Compute the F1 and exact scores.
	results = squad_evaluate(examples, predictions)
	return results

def evaluateNLI(model, data, device):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		matrix = [[0 for _ in range(3)] for _ in range(3)]
		for j,batch in enumerate(data):
			batch['label'] = batch['label'].to(device)
			output = model.forward('sc',batch)
			loss, logits = output[0].mean(), output[1]
			prediction = torch.argmax(logits, dim=1)
			correct += torch.sum(prediction == batch['label']).item()
			for k in range(batch['label'].shape[0]):
				matrix[batch['label'][k]][prediction[k]] += 1
			total += batch['label'].shape[0]
			total_loss += loss.item()

		total_loss /= len(data)

		return total_loss, correct / total#, matrix
	
def evaluatePA(model, data, device):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		matrix = [[0 for _ in range(2)] for _ in range(2)]
		for j,batch in enumerate(data):
			batch['label'] = batch['label'].to(device)
			output = model.forward('pa',batch)
			loss, logits = output[0].mean(), output[1]
			prediction = torch.argmax(logits, dim=1)
			correct += torch.sum(prediction == batch['label']).item()
			for k in range(batch['label'].shape[0]):
				matrix[batch['label'][k]][prediction[k]] += 1
			total += batch['label'].shape[0]
			total_loss += loss.item()

		total_loss /= len(data)

		return total_loss, correct / total#, matrix

def macro_f1(array):
	n = len(array)
	p = [0 for _ in range(n-1)]
	r = [0 for _ in range(n-1)]
	f1 = [0 for _ in range(n-1)]
	for i in range(n-1):
		p[i] = array[i][i] / sum([array[j][i] for j in range(n-1)]) if sum([array[j][i] for j in range(n-1)]) != 0 else 0
		r[i] = array[i][i] / sum([array[i][j] for j in range(n-1)]) if sum([array[i][j] for j in range(n-1)]) != 0 else 0
		f1[i] = 2*p[i]*r[i] / (p[i] + r[i]) if (p[i] + r[i] != 0) else 0
	return sum(f1) / (n-1)

def evaluateNER(model, data, device):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		matrix = [[0 for _ in range(10)] for _ in range(10)]
		for j,batch in enumerate(data):
			batch['mask'] = batch['mask'].to(device)
			batch['label_ids'] = batch['label_ids'].to(device)
			output = model.forward('tc',batch)
			loss, logits = output[0].mean(), output[1]
			active_loss = batch['mask'].view(-1) == 1
			active_logits = logits.view(-1, 10)[active_loss]
			active_labels = batch['label_ids'].view(-1)[active_loss]
			prediction = torch.argmax(active_logits, dim=1)
			correct += torch.sum(prediction == active_labels).item()
			for k in range(active_labels.shape[0]):
				matrix[active_labels[k]][prediction[k]] += 1
			total += active_labels.shape[0]
			total_loss += loss.item()

		total_loss /= len(data)

		return total_loss, correct / total# macro_f1(matrix)

def evaluatePOS(model, data, device):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		for j,batch in enumerate(data):
			batch['mask'] = batch['mask'].to(device)
			batch['label_ids'] = batch['label_ids'].to(device)
			output = model.forward('po',batch)
			loss, logits = output[0].mean(), output[1]
			active_loss = batch['mask'].view(-1) == 1
			active_logits = logits.view(-1, 18)[active_loss]
			active_labels = batch['label_ids'].view(-1)[active_loss]
			prediction = torch.argmax(active_logits, dim=1)
			correct += torch.sum(prediction == active_labels).item()
			total += active_labels.shape[0]
			total_loss += loss.item()

		total_loss /= len(data)

		return total_loss, correct / total

def evaluateRC(model,data,device):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		
		for j,batch in enumerate(data):
			batch['answer_tok_pos'] = batch['answer_tok_pos'].to(device)
			output = model.forward('rc',batch)
			loss, logits = output[0].mean(), output[1]
			prediction = torch.argmax(logits, dim=1)
			correct += torch.sum(prediction == batch['answer_tok_pos']).item()
			total += batch['answer_tok_pos'].shape[0]
			total_loss += loss.item()

		total_loss /= len(data)

		return total_loss, correct / total
