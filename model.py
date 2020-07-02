import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel

class BertMetaLearning(nn.Module):

	def __init__(self, args):
		super(BertMetaLearning,self).__init__()
		self.args = args
		self.device = None

		self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

		# Question Answering
		self.qa_outputs = nn.Linear(args.hidden_dims, args.qa_labels)

		# # Token Classification (Named Entity Recognition)
		self.tc_dropout = nn.Dropout(args.dropout)
		self.tc_classifier = nn.Linear(args.hidden_dims, args.tc_labels)

		# # Token Classification (Part of Speech Tagging)
		self.po_dropout = nn.Dropout(args.dropout)
		self.po_classifier = nn.Linear(args.hidden_dims, args.po_labels)

		# Sequence Classification
		self.sc_dropout = nn.Dropout(args.dropout)
		self.sc_classifier = nn.Linear(args.hidden_dims, args.sc_labels)

		# Paraphrase
		self.pa_dropout = nn.Dropout(args.dropout)
		self.pa_classifier = nn.Linear(args.hidden_dims, args.pa_labels)

	def forward(self, task, data):

		if 'qa' in task:

			data['input_ids'] = data['input_ids'].to(self.device)
			data['attention_mask'] = data['attention_mask'].to(self.device)
			data['token_type_ids'] = data['token_type_ids'].to(self.device)

			outputs = self.bert(data['input_ids'],
								attention_mask=data['attention_mask'],
								token_type_ids=data['token_type_ids'])

			batch_size = data['input_ids'].shape[0]
			sequence_output = outputs[0]

			logits = self.qa_outputs(sequence_output)
			start_logits, end_logits = logits.split(1, dim=-1)
			start_logits = start_logits.squeeze(-1)
			end_logits = end_logits.squeeze(-1)
			outputs = (start_logits, end_logits,) + outputs[2:]

			if data['answer_start'] is not None and data['answer_end'] is not None:
				data['answer_start'] = data['answer_start'].to(self.device)
				data['answer_end'] = data['answer_end'].to(self.device)
				ignored_index = start_logits.size(1)
				data['answer_start'].clamp_(0, ignored_index)
				data['answer_end'].clamp_(0, ignored_index)
				start_loss = F.cross_entropy(start_logits, data['answer_start'], ignore_index=ignored_index,reduction='none')
				end_loss = F.cross_entropy(end_logits, data['answer_end'], ignore_index=ignored_index,reduction='none')
				loss = (start_loss + end_loss) / 2
				outputs = (loss, ) + outputs

		elif 'sc' in task:

			data['input_ids'] = data['input_ids'].to(self.device)
			data['attention_mask'] = data['attention_mask'].to(self.device)
			data['token_type_ids'] = data['token_type_ids'].to(self.device)
			data['label'] = data['label'].to(self.device)

			outputs = self.bert(data['input_ids'],
								attention_mask=data['attention_mask'],
								token_type_ids=data['token_type_ids'])

			batch_size = data['input_ids'].shape[0]
			pooled_output = outputs[1]

			pooled_output = self.sc_dropout(pooled_output)
			logits = self.sc_classifier(pooled_output)

			loss = F.cross_entropy(logits, data['label'],reduction='none')
			outputs = (loss, logits) + outputs[2:]

		elif 'pa' in task:

			data['input_ids'] = data['input_ids'].to(self.device)
			data['attention_mask'] = data['attention_mask'].to(self.device)
			data['token_type_ids'] = data['token_type_ids'].to(self.device)
			data['label'] = data['label'].to(self.device)

			outputs = self.bert(data['input_ids'],
								attention_mask=data['attention_mask'],
								token_type_ids=data['token_type_ids'])

			batch_size = data['input_ids'].shape[0]
			pooled_output = outputs[1]

			pooled_output = self.pa_dropout(pooled_output)
			logits = self.pa_classifier(pooled_output)

			loss = F.cross_entropy(logits, data['label'],reduction='none')
			outputs = (loss, logits) + outputs[2:]

		elif 'tc' in task:

			data['input_ids'] = data['input_ids'].to(self.device)
			data['attention_mask'] = data['attention_mask'].to(self.device)
			data['token_type_ids'] = data['token_type_ids'].to(self.device)
			data['mask'] = data['mask'].float().to(self.device)
			data['label_ids'] = data['label_ids'].to(self.device)

			outputs = self.bert(data['input_ids'],
								attention_mask=data['attention_mask'],
								token_type_ids=data['token_type_ids'])

			batch_size = data['input_ids'].shape[0]
			sequence_output = outputs[0]

			sequence_output = self.tc_dropout(sequence_output)
			logits = self.tc_classifier(sequence_output).view(-1, self.args.tc_labels)
			labels = data['label_ids'].view(-1)
			loss = F.cross_entropy(logits, labels,reduction='none')
			loss = loss*data['mask'].view(-1)
			loss = loss.view(batch_size,-1)
			loss = loss.sum(dim=1)/data['mask'].sum(dim=1)
			outputs = (loss, logits) + outputs[2:]
		
		elif 'po' in task:

			data['input_ids'] = data['input_ids'].to(self.device)
			data['attention_mask'] = data['attention_mask'].to(self.device)
			data['token_type_ids'] = data['token_type_ids'].to(self.device)
			data['mask'] = data['mask'].float().to(self.device)
			data['label_ids'] = data['label_ids'].to(self.device)

			outputs = self.bert(data['input_ids'],
								attention_mask=data['attention_mask'],
								token_type_ids=data['token_type_ids'])

			batch_size = data['input_ids'].shape[0]
			sequence_output = outputs[0]

			sequence_output = self.po_dropout(sequence_output)
			logits = self.po_classifier(sequence_output).view(-1, self.args.po_labels)
			labels = data['label_ids'].view(-1)
			loss = F.cross_entropy(logits, labels,reduction='none')
			loss = loss*data['mask'].view(-1)
			loss = loss.view(batch_size,-1)
			loss = loss.sum(dim=1)/data['mask'].sum(dim=1)
			outputs = (loss, logits) + outputs[2:]

		return outputs

	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		self.bert = self.bert.to(*args, **kwargs)
		self.qa_outputs = self.qa_outputs.to(*args, **kwargs)
		self.tc_dropout = self.tc_dropout.to(*args, **kwargs)
		self.tc_classifier = self.tc_classifier.to(*args, **kwargs)
		self.po_dropout = self.po_dropout.to(*args, **kwargs)
		self.po_classifier = self.po_classifier.to(*args, **kwargs)
		self.sc_dropout = self.sc_dropout.to(*args, **kwargs)
		self.sc_classifier = self.sc_classifier.to(*args, **kwargs)
		self.pa_dropout = self.pa_dropout.to(*args, **kwargs)
		self.pa_classifier = self.pa_classifier.to(*args, **kwargs)
		return self
