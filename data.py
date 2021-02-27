import torch
from transformers import BertTokenizer, BertModel
from termcolor import colored
from tqdm import tqdm
import numpy as np
import json, os, sys
import pickle,csv
import collections
from torch.utils.data import Dataset
import collections
# from bert import tokenization
import string
from helper import *
import re

class CorpusQA(Dataset):

	def __init__(self, path, evaluate):
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'
		self.pad_token = 0
		self.sequence_a_segment_id = 0
		self.sequence_b_segment_id = 1
		self.pad_token_segment_id = 0
		self.cls_token_segment_id = 0
		self.mask_padding_with_zero = True
		self.doc_stride = 128
		self.max_query_len = 64
		self.max_seq_len = 384

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)

		self.dataset, self.examples, self.features = self.preprocess(path, evaluate)

	def preprocess(self, file, evaluate=False):

		fname = file.split('/')[-1]
		cached_features_file = os.path.join('/'.join(file.split('/')[:-1]),"cached_{}".format(fname))

		# Init features and dataset from cache if it exists
		if os.path.exists(cached_features_file):
			features_and_dataset = torch.load(cached_features_file)
			features, dataset, examples = (
				features_and_dataset["features"],
				features_and_dataset["dataset"],
				features_and_dataset["examples"],
			)
		else:

			processor = SquadProcessor()
			if evaluate:
				examples = processor.get_dev_examples(file)
			else:
				examples = processor.get_train_examples(file)

			features, dataset = squad_convert_examples_to_features(
				examples=examples,
				tokenizer=self.tokenizer,
				max_seq_length=self.max_seq_len,
				doc_stride=self.doc_stride,
				max_query_length=self.max_query_len,
				is_training=not evaluate,
				return_dataset="pt",
				threads=1,
			)

			torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

		return dataset, examples, features

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, id):
		return {
				"input_ids": self.dataset[id][0],
				"attention_mask": self.dataset[id][1],
				"token_type_ids": self.dataset[id][2],
				"answer_start": self.dataset[id][3],
				"answer_end": self.dataset[id][4],
			}

class CorpusSC(Dataset):

	def __init__(self, path, file):
		self.max_sequence_length = 128
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'
		self.pad_token = 0
		self.sequence_a_segment_id = 0
		self.sequence_b_segment_id = 1
		self.pad_token_segment_id = 0
		self.cls_token_segment_id = 0
		self.mask_padding_with_zero = True
		self.doc_stride = 128
		self.max_query_length = 64

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)

		self.labels = {'contradiction':0, 'entailment':1, 'neutral':2}

		if os.path.exists(path + '.pickle'):
			self.data = pickle.load(open(path + '.pickle','rb'))
		else:
			self.data = self.preprocess(path, file)
			with open(path + '.pickle','wb') as f:
				pickle.dump(self.data,f,protocol=pickle.HIGHEST_PROTOCOL)

	def preprocess(self, path, file):

		list_label = []
		list_input_ids = []
		list_token_type_ids = []
		list_attention_mask = []

		if file == 'csv':
			with open(path, encoding="utf-8") as f:
				reader = csv.reader(f, delimiter="\t")
				for line in reader:
					try:
						label = line[2]
					except:
						print (line)
					sentence1_tokenized = self.tokenizer.tokenize(line[0])
					sentence2_tokenized = self.tokenizer.tokenize(line[1])

					if len(sentence1_tokenized) + len(sentence2_tokenized) + 3 > self.max_sequence_length:
						continue

					input_ids, token_type_ids, attention_mask = self.encode(sentence1_tokenized, sentence2_tokenized)
					list_label.append(self.labels[label])
					list_input_ids.append(torch.unsqueeze(input_ids,dim=0))
					list_token_type_ids.append(torch.unsqueeze(token_type_ids,dim=0))
					list_attention_mask.append(torch.unsqueeze(attention_mask,dim=0))
		else:
			with open(path, encoding="utf-8") as f:
				data = [json.loads(jline) for jline in f.readlines()]
				for row in tqdm(data):
					label = row["gold_label"]

					if label not in ['neutral','contradiction','entailment']: continue

					sentence1_tokenized = self.tokenizer.tokenize(row["sentence1"])
					sentence2_tokenized = self.tokenizer.tokenize(row["sentence2"])
					if len(sentence1_tokenized) + len(sentence2_tokenized) + 3 > self.max_sequence_length:
						continue

					input_ids, token_type_ids, attention_mask = self.encode(sentence1_tokenized, sentence2_tokenized)
					list_label.append(self.labels[label])
					list_input_ids.append(torch.unsqueeze(input_ids,dim=0))
					list_token_type_ids.append(torch.unsqueeze(token_type_ids,dim=0))
					list_attention_mask.append(torch.unsqueeze(attention_mask,dim=0))

		list_label = torch.tensor(list_label)
		list_input_ids = torch.cat(list_input_ids,dim = 0)
		list_token_type_ids = torch.cat(list_token_type_ids, dim=0)
		list_attention_mask = torch.cat(list_attention_mask, dim=0)

		dataset = {
			'input_ids':list_input_ids,
			'token_type_ids':list_token_type_ids,
			'attention_mask':list_attention_mask,
			'label':list_label,
		}

		return dataset

	def encode(self, sentence1, sentence2):

		tokens = []
		segment_mask = []
		input_mask = []

		tokens.append(self.cls_token)
		segment_mask.append(self.cls_token_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		for tok in sentence1:
			tokens.append(tok)
			segment_mask.append(self.sequence_a_segment_id)
			input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens.append(self.sep_token)
		segment_mask.append(self.sequence_a_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		for tok in sentence2:
			tokens.append(tok)
			segment_mask.append(self.sequence_b_segment_id)
			input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens.append(self.sep_token)
		segment_mask.append(self.sequence_b_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens = self.tokenizer.convert_tokens_to_ids(tokens)

		while(len(tokens) < self.max_sequence_length):
			tokens.append(self.pad_token)
			segment_mask.append(self.pad_token_segment_id)
			input_mask.append(0 if self.mask_padding_with_zero else 1)

		tokens = torch.tensor(tokens)
		segment_mask = torch.tensor(segment_mask)
		input_mask = torch.tensor(input_mask)

		return tokens, segment_mask, input_mask

	def __len__(self):
		return self.data['input_ids'].shape[0]

	def __getitem__(self, id):

		return {
			'input_ids':self.data['input_ids'][id],
			'token_type_ids':self.data['token_type_ids'][id],
			'attention_mask':self.data['attention_mask'][id],
			'label':self.data['label'][id],
		}

class CorpusPO(Dataset):

	def __init__(self, path):
		self.max_sequence_length = 128
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'
		self.pad_token = 0
		self.mask_padding_with_zero = True

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
		self.labels_list = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "BLK"]

		if os.path.exists(path + '.pickle'):
			self.data = pickle.load(open(path + '.pickle','rb'))
		else:
			self.data = self.preprocess(path)
			with open(path + '.pickle','wb') as f:
				pickle.dump(self.data,f,protocol=pickle.HIGHEST_PROTOCOL)

	def preprocess(self, file):

		list_input_ids, list_input_mask, list_segment_ids, list_labels, list_mask = [], [], [], [], []

		dataset = []
		with open(file,'r',encoding='utf8') as f:
			sentence = []
			label= []
			for i,line in enumerate(f.readlines()):
				if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
					if len(sentence) > 0 and len(sentence):
						dataset.append((sentence,label))
						sentence = []
						label = []
					continue
				splits = line.strip().split('\t')
				if len(splits) == 1:
					continue
				sentence.append(splits[0])
				label.append(splits[1])

			if len(sentence) > 0 and len(sentence):
				dataset.append((sentence,label))
				sentence = []
				label = []

		print (len(dataset))
		label_map = {label : i for i, label in enumerate(self.labels_list)}

		for i, example in enumerate(tqdm(dataset)):

			tokens = [self.cls_token]
			segment_ids = [0]
			input_mask = [1]
			labels = [label_map['BLK']]
			label_mask = [0]

			for j, w in enumerate(example[0]):
				token = self.tokenizer.tokenize(w)
				tokens.extend(token)
				label = example[1][j]
				for m in range(len(token)):
					segment_ids.append(0)
					input_mask.append(1)
					if m == 0:
						labels.append(label_map[label])
						label_mask.append(1)
					else:
						labels.append(label_map['BLK'])
						label_mask.append(0)

			if len(tokens) > self.max_sequence_length - 1:
				tokens = tokens[0:(self.max_sequence_length - 1)]
				segment_ids = segment_ids[0:(self.max_sequence_length - 1)]
				input_mask = input_mask[0:(self.max_sequence_length) - 1]
				labels = labels[0:(self.max_sequence_length - 1)]
				label_mask = label_mask[0:(self.max_sequence_length - 1)]

			tokens.append(self.sep_token)
			segment_ids.append(0)
			input_mask.append(1)
			label_mask.append(0)
			labels.append(label_map['BLK'])

			input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

			while len(input_ids) < self.max_sequence_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
				labels.append(0)
				label_mask.append(0)

			assert len(input_ids) == self.max_sequence_length
			assert len(input_mask) == self.max_sequence_length
			assert len(segment_ids) == self.max_sequence_length
			assert len(labels) == self.max_sequence_length
			assert len(label_mask) == self.max_sequence_length

			list_input_ids.append(torch.unsqueeze(torch.tensor(input_ids),0))
			list_input_mask.append(torch.unsqueeze(torch.tensor(input_mask),0))
			list_segment_ids.append(torch.unsqueeze(torch.tensor(segment_ids),0))
			list_labels.append(torch.unsqueeze(torch.tensor(labels),0))
			list_mask.append(torch.unsqueeze(torch.tensor(label_mask),0))

		list_input_ids = torch.cat(list_input_ids,dim=0)
		list_input_mask = torch.cat(list_input_mask,dim=0)
		list_segment_ids = torch.cat(list_segment_ids,dim=0)
		list_labels = torch.cat(list_labels,dim=0)
		list_mask = torch.cat(list_mask,dim=0)

		dataset = {
			'input_ids':list_input_ids,
			'attention_mask':list_input_mask,
			'token_type_ids':list_segment_ids,
			'label_ids':list_labels,
			'mask':list_mask,
		}

		return dataset

	def __len__(self):
		return self.data['input_ids'].shape[0]

	def __getitem__(self, id):
		return {
			'input_ids':self.data['input_ids'][id],
			'token_type_ids':self.data['token_type_ids'][id],
			'attention_mask':self.data['attention_mask'][id],
			'label_ids':self.data['label_ids'][id],
			'mask':self.data['mask'][id],
		}


class CorpusTC(Dataset):

	def __init__(self, path):
		self.max_sequence_length = 128
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'
		self.pad_token = 0
		self.mask_padding_with_zero = True

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
		self.labels_list = ["O", "B-PER", "B-LOC", "B-ORG", "B-MISC", "I-PER", "I-LOC", "I-ORG", "I-MISC", "BLK"]

		if os.path.exists(path + '.pickle'):
			self.data = pickle.load(open(path + '.pickle','rb'))
		else:
			self.data = self.preprocess(path)
			with open(path + '.pickle','wb') as f:
				pickle.dump(self.data,f,protocol=pickle.HIGHEST_PROTOCOL)

	def preprocess(self, file):

		list_input_ids, list_input_mask, list_segment_ids, list_labels, list_mask = [], [], [], [], []

		dataset = []
		with open(file,'r',encoding='utf8') as f:
			sentence = []
			label= []
			for i,line in enumerate(f.readlines()):
				if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
					if len(sentence) > 0 and len(sentence):
						dataset.append((sentence,label))
						sentence = []
						label = []
					continue
				splits = line.strip().split('\t')
				if len(splits) == 1:
					continue
				sentence.append(splits[0])
				label.append(splits[1])

			if len(sentence) > 0 and len(sentence):
				dataset.append((sentence,label))
				sentence = []
				label = []

		print (len(dataset))
		label_map = {label : i for i, label in enumerate(self.labels_list)}

		for i, example in enumerate(tqdm(dataset)):

			tokens = [self.cls_token]
			segment_ids = [0]
			input_mask = [1]
			labels = [label_map['BLK']]
			label_mask = [0]

			for j, w in enumerate(example[0]):
				token = self.tokenizer.tokenize(w)
				tokens.extend(token)
				label = example[1][j]
				for m in range(len(token)):
					segment_ids.append(0)
					input_mask.append(1)
					if m == 0:
						labels.append(label_map[label])
						label_mask.append(1)
					else:
						labels.append(label_map['BLK'])
						label_mask.append(0)

			if len(tokens) > self.max_sequence_length - 1:
				tokens = tokens[0:(self.max_sequence_length - 1)]
				segment_ids = segment_ids[0:(self.max_sequence_length - 1)]
				input_mask = input_mask[0:(self.max_sequence_length) - 1]
				labels = labels[0:(self.max_sequence_length - 1)]
				label_mask = label_mask[0:(self.max_sequence_length - 1)]

			tokens.append(self.sep_token)
			segment_ids.append(0)
			input_mask.append(1)
			label_mask.append(0)
			labels.append(label_map['BLK'])

			input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

			while len(input_ids) < self.max_sequence_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
				labels.append(0)
				label_mask.append(0)

			assert len(input_ids) == self.max_sequence_length
			assert len(input_mask) == self.max_sequence_length
			assert len(segment_ids) == self.max_sequence_length
			assert len(labels) == self.max_sequence_length
			assert len(label_mask) == self.max_sequence_length

			list_input_ids.append(torch.unsqueeze(torch.tensor(input_ids),0))
			list_input_mask.append(torch.unsqueeze(torch.tensor(input_mask),0))
			list_segment_ids.append(torch.unsqueeze(torch.tensor(segment_ids),0))
			list_labels.append(torch.unsqueeze(torch.tensor(labels),0))
			list_mask.append(torch.unsqueeze(torch.tensor(label_mask),0))

		list_input_ids = torch.cat(list_input_ids,dim=0)
		list_input_mask = torch.cat(list_input_mask,dim=0)
		list_segment_ids = torch.cat(list_segment_ids,dim=0)
		list_labels = torch.cat(list_labels,dim=0)
		list_mask = torch.cat(list_mask,dim=0)

		dataset = {
			'input_ids':list_input_ids,
			'attention_mask':list_input_mask,
			'token_type_ids':list_segment_ids,
			'label_ids':list_labels,
			'mask':list_mask,
		}

		return dataset

	def __len__(self):
		return self.data['input_ids'].shape[0]

	def __getitem__(self, id):
		return {
			'input_ids':self.data['input_ids'][id],
			'token_type_ids':self.data['token_type_ids'][id],
			'attention_mask':self.data['attention_mask'][id],
			'label_ids':self.data['label_ids'][id],
			'mask':self.data['mask'][id],
		}

class CorpusPA(Dataset):

	def __init__(self, path):
		self.max_sequence_length = 128
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'
		self.pad_token = 0
		self.sequence_a_segment_id = 0
		self.sequence_b_segment_id = 1
		self.pad_token_segment_id = 0
		self.cls_token_segment_id = 0
		self.mask_padding_with_zero = True
		self.doc_stride = 128
		self.max_query_length = 64

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)

		if os.path.exists(path + '.pickle'):
			self.data = pickle.load(open(path + '.pickle','rb'))
		else:
			self.data = self.preprocess(path)
			with open(path + '.pickle','wb') as f:
				pickle.dump(self.data,f,protocol=pickle.HIGHEST_PROTOCOL)

	def preprocess(self, path):

		list_label = []
		list_input_ids = []
		list_token_type_ids = []
		list_attention_mask = []

		with open(path, encoding="utf-8") as f:
			lines = f.readlines()
			for i,line in enumerate(lines):
				line = line.strip().split('\t')
				try:
					label = int(line[2])
					sentence1_tokenized = self.tokenizer.tokenize(line[0])
					sentence2_tokenized = self.tokenizer.tokenize(line[1])
				except:
					continue

				if len(sentence1_tokenized) + len(sentence2_tokenized) + 3 > self.max_sequence_length:
					continue

				input_ids, token_type_ids, attention_mask = self.encode(sentence1_tokenized, sentence2_tokenized)
				list_label.append(label)
				list_input_ids.append(torch.unsqueeze(input_ids,dim=0))
				list_token_type_ids.append(torch.unsqueeze(token_type_ids,dim=0))
				list_attention_mask.append(torch.unsqueeze(attention_mask,dim=0))

		list_label = torch.tensor(list_label)
		list_input_ids = torch.cat(list_input_ids,dim = 0)
		list_token_type_ids = torch.cat(list_token_type_ids, dim=0)
		list_attention_mask = torch.cat(list_attention_mask, dim=0)

		dataset = {
			'input_ids':list_input_ids,
			'token_type_ids':list_token_type_ids,
			'attention_mask':list_attention_mask,
			'label':list_label,
		}

		return dataset

	def encode(self, sentence1, sentence2):

		tokens = []
		segment_mask = []
		input_mask = []

		tokens.append(self.cls_token)
		segment_mask.append(self.cls_token_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		for tok in sentence1:
			tokens.append(tok)
			segment_mask.append(self.sequence_a_segment_id)
			input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens.append(self.sep_token)
		segment_mask.append(self.sequence_a_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		for tok in sentence2:
			tokens.append(tok)
			segment_mask.append(self.sequence_b_segment_id)
			input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens.append(self.sep_token)
		segment_mask.append(self.sequence_b_segment_id)
		input_mask.append(1 if self.mask_padding_with_zero else 0)

		tokens = self.tokenizer.convert_tokens_to_ids(tokens)

		while(len(tokens) < self.max_sequence_length):
			tokens.append(self.pad_token)
			segment_mask.append(self.pad_token_segment_id)
			input_mask.append(0 if self.mask_padding_with_zero else 1)

		tokens = torch.tensor(tokens)
		segment_mask = torch.tensor(segment_mask)
		input_mask = torch.tensor(input_mask)

		return tokens, segment_mask, input_mask

	def __len__(self):
		return self.data['input_ids'].shape[0]

	def __getitem__(self, id):

		return {
			'input_ids':self.data['input_ids'][id],
			'token_type_ids':self.data['token_type_ids'][id],
			'attention_mask':self.data['attention_mask'][id],
			'label':self.data['label'][id],
		}
