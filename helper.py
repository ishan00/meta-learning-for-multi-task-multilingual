import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

import copy
import csv
import json

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class InputExample(object):
	"""
	A single training/test example for simple sequence classification.

	Args:
		guid: Unique id for the example.
		text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
		text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
		label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
	"""

	def __init__(self, guid, text_a, text_b=None, label=None):
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
	"""
	A single set of features of data.

	Args:
		input_ids: Indices of input sequence tokens in the vocabulary.
		attention_mask: Mask to avoid performing attention on padding token indices.
			Mask values selected in ``[0, 1]``:
			Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
		token_type_ids: Segment token indices to indicate first and second portions of the inputs.
		label: Label corresponding to the input
	"""

	def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.token_type_ids = token_type_ids
		self.label = label

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_example_from_tensor_dict(self, tensor_dict):
		"""Gets an example from a dict with tensorflow tensors
		Args:
			tensor_dict: Keys and values should match the corresponding Glue
				tensorflow_dataset examples.
		"""
		raise NotImplementedError()

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	def tfds_map(self, example):
		"""Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
		This method converts examples to the correct format."""
		if len(self.get_labels()) > 1:
			example.label = self.get_labels()[int(example.label)]
		return example

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8-sig") as f:
			return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class SingleSentenceClassificationProcessor(DataProcessor):
	""" Generic processor for a single sentence classification data set."""

	def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
		self.labels = [] if labels is None else labels
		self.examples = [] if examples is None else examples
		self.mode = mode
		self.verbose = verbose

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
		return self.examples[idx]

	@classmethod
	def create_from_csv(
		cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
	):
		processor = cls(**kwargs)
		processor.add_examples_from_csv(
			file_name,
			split_name=split_name,
			column_label=column_label,
			column_text=column_text,
			column_id=column_id,
			skip_first_row=skip_first_row,
			overwrite_labels=True,
			overwrite_examples=True,
		)
		return processor

	@classmethod
	def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
		processor = cls(**kwargs)
		processor.add_examples(texts_or_text_and_labels, labels=labels)
		return processor

	def add_examples_from_csv(
		self,
		file_name,
		split_name="",
		column_label=0,
		column_text=1,
		column_id=None,
		skip_first_row=False,
		overwrite_labels=False,
		overwrite_examples=False,
	):
		lines = self._read_tsv(file_name)
		if skip_first_row:
			lines = lines[1:]
		texts = []
		labels = []
		ids = []
		for (i, line) in enumerate(lines):
			texts.append(line[column_text])
			labels.append(line[column_label])
			if column_id is not None:
				ids.append(line[column_id])
			else:
				guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
				ids.append(guid)

		return self.add_examples(
			texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
		)

	def add_examples(
		self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
	):
		assert labels is None or len(texts_or_text_and_labels) == len(labels)
		assert ids is None or len(texts_or_text_and_labels) == len(ids)
		if ids is None:
			ids = [None] * len(texts_or_text_and_labels)
		if labels is None:
			labels = [None] * len(texts_or_text_and_labels)
		examples = []
		added_labels = set()
		for (text_or_text_and_label, label, guid) in zip(texts_or_text_and_labels, labels, ids):
			if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
				text, label = text_or_text_and_label
			else:
				text = text_or_text_and_label
			added_labels.add(label)
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

		# Update examples
		if overwrite_examples:
			self.examples = examples
		else:
			self.examples.extend(examples)

		# Update labels
		if overwrite_labels:
			self.labels = list(added_labels)
		else:
			self.labels = list(set(self.labels).union(added_labels))

		return self.examples

	def get_features(
		self,
		tokenizer,
		max_length=None,
		pad_on_left=False,
		pad_token=0,
		mask_padding_with_zero=True,
		return_tensors=None,
	):
		"""
		Convert examples in a list of ``InputFeatures``

		Args:
			tokenizer: Instance of a tokenizer that will tokenize the examples
			max_length: Maximum example length
			task: GLUE task
			label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
			output_mode: String indicating the output mode. Either ``regression`` or ``classification``
			pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
			pad_token: Padding token
			mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
				and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
				actual values)

		Returns:
			If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
			containing the task-specific features. If the input is a list of ``InputExamples``, will return
			a list of task-specific ``InputFeatures`` which can be fed to the model.

		"""
		if max_length is None:
			max_length = tokenizer.max_len

		label_map = {label: i for i, label in enumerate(self.labels)}

		all_input_ids = []
		for (ex_index, example) in enumerate(self.examples):
			#if ex_index % 10000 == 0:
			#    logger.info("Tokenizing example %d", ex_index)

			input_ids = tokenizer.encode(
				example.text_a, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
			)
			all_input_ids.append(input_ids)

		batch_length = max(len(input_ids) for input_ids in all_input_ids)

		features = []
		for (ex_index, (input_ids, example)) in enumerate(zip(all_input_ids, self.examples)):
			#if ex_index % 10000 == 0:
			#    logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))
			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

			# Zero-pad up to the sequence length.
			padding_length = batch_length - len(input_ids)
			if pad_on_left:
				input_ids = ([pad_token] * padding_length) + input_ids
				attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
			else:
				input_ids = input_ids + ([pad_token] * padding_length)
				attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

			assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
				len(input_ids), batch_length
			)
			assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
				len(attention_mask), batch_length
			)

			if self.mode == "classification":
				label = label_map[example.label]
			elif self.mode == "regression":
				label = float(example.label)
			else:
				raise ValueError(self.mode)

			#if ex_index < 5 and self.verbose:
			#    logger.info("*** Example ***")
			#    logger.info("guid: %s" % (example.guid))
			#    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			#    logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
			#    logger.info("label: %s (id = %d)" % (example.label, label))

			features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label=label))

		if return_tensors is None:
			return features
		elif return_tensors == "tf":
			if not is_tf_available():
				raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
			import tensorflow as tf

			def gen():
				for ex in features:
					yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

			dataset = tf.data.Dataset.from_generator(
				gen,
				({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
				({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
			)
			return dataset
		elif return_tensors == "pt":
			if not is_torch_available():
				raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
			import torch
			from torch.utils.data import TensorDataset

			all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
			all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
			if self.mode == "classification":
				all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
			elif self.mode == "regression":
				all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

			dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
			return dataset
		else:
			raise ValueError("return_tensors should be one of 'tf' or 'pt'")

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
	"""Returns tokenized answer spans that better match the annotated answer."""
	tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

	for new_start in range(input_start, input_end + 1):
		for new_end in range(input_end, new_start - 1, -1):
			text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
			if text_span == tok_answer_text:
				return (new_start, new_end)

	return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span.start + doc_span.length - 1
		if position < doc_span.start:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span.start
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index

	return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""
	# if len(doc_spans) == 1:
	# return True
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span["start"] + doc_span["length"] - 1
		if position < doc_span["start"]:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span["start"]
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index

	return cur_span_index == best_span_index


def _is_whitespace(c):
	if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
		return True
	return False


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
	features = []
	if is_training and not example.is_impossible:
		# Get start and end position
		start_position = example.start_position
		end_position = example.end_position

		# If the answer cannot be found in the text, then skip this example.
		actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
		cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
		if actual_text.find(cleaned_answer_text) == -1:
			#logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
			return []

	tok_to_orig_index = []
	orig_to_tok_index = []
	all_doc_tokens = []
	for (i, token) in enumerate(example.doc_tokens):
		orig_to_tok_index.append(len(all_doc_tokens))
		sub_tokens = tokenizer.tokenize(token)
		for sub_token in sub_tokens:
			tok_to_orig_index.append(i)
			all_doc_tokens.append(sub_token)

	if is_training and not example.is_impossible:
		tok_start_position = orig_to_tok_index[example.start_position]
		if example.end_position < len(example.doc_tokens) - 1:
			tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
		else:
			tok_end_position = len(all_doc_tokens) - 1

		(tok_start_position, tok_end_position) = _improve_answer_span(
			all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
		)

	spans = []
	truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
	sequence_added_tokens = (
		tokenizer.max_len - tokenizer.max_len_single_sentence + 1
		if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
		else tokenizer.max_len - tokenizer.max_len_single_sentence
	)
	sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

	span_doc_tokens = all_doc_tokens
	while len(spans) * doc_stride < len(all_doc_tokens):

		encoded_dict = tokenizer.encode_plus(
			truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
			span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
			max_length=max_seq_length,
			return_overflowing_tokens=True,
			pad_to_max_length=True,
			stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
			truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
		)

		paragraph_len = min(
			len(all_doc_tokens) - len(spans) * doc_stride,
			max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
		)

		if tokenizer.pad_token_id in encoded_dict["input_ids"]:
			if tokenizer.padding_side == "right":
				non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
			else:
				last_padding_id_position = (
					len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
				)
				non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

		else:
			non_padded_ids = encoded_dict["input_ids"]

		tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

		token_to_orig_map = {}
		for i in range(paragraph_len):
			index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
			token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

		encoded_dict["paragraph_len"] = paragraph_len
		encoded_dict["tokens"] = tokens
		encoded_dict["token_to_orig_map"] = token_to_orig_map
		encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
		encoded_dict["token_is_max_context"] = {}
		encoded_dict["start"] = len(spans) * doc_stride
		encoded_dict["length"] = paragraph_len

		spans.append(encoded_dict)

		if "overflowing_tokens" not in encoded_dict:
			break
		span_doc_tokens = encoded_dict["overflowing_tokens"]

	for doc_span_index in range(len(spans)):
		for j in range(spans[doc_span_index]["paragraph_len"]):
			is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
			index = (
				j
				if tokenizer.padding_side == "left"
				else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
			)
			spans[doc_span_index]["token_is_max_context"][index] = is_max_context

	for span in spans:
		# Identify the position of the CLS token
		cls_index = span["input_ids"].index(tokenizer.cls_token_id)

		# p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
		# Original TF implem also keep the classification token (set to 0) (not sure why...)
		p_mask = np.array(span["token_type_ids"])

		p_mask = np.minimum(p_mask, 1)

		if tokenizer.padding_side == "right":
			# Limit positive values to one
			p_mask = 1 - p_mask

		p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

		# Set the CLS index to '0'
		p_mask[cls_index] = 0

		span_is_impossible = example.is_impossible
		start_position = 0
		end_position = 0
		if is_training and not span_is_impossible:
			# For training, if our document chunk does not contain an annotation
			# we throw it out, since there is nothing to predict.
			doc_start = span["start"]
			doc_end = span["start"] + span["length"] - 1
			out_of_span = False

			if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
				out_of_span = True

			if out_of_span:
				start_position = cls_index
				end_position = cls_index
				span_is_impossible = True
			else:
				if tokenizer.padding_side == "left":
					doc_offset = 0
				else:
					doc_offset = len(truncated_query) + sequence_added_tokens

				start_position = tok_start_position - doc_start + doc_offset
				end_position = tok_end_position - doc_start + doc_offset

		features.append(
			SquadFeatures(
				span["input_ids"],
				span["attention_mask"],
				span["token_type_ids"],
				cls_index,
				p_mask.tolist(),
				example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
				unique_id=0,
				paragraph_len=span["paragraph_len"],
				token_is_max_context=span["token_is_max_context"],
				tokens=span["tokens"],
				token_to_orig_map=span["token_to_orig_map"],
				start_position=start_position,
				end_position=end_position,
				is_impossible=span_is_impossible,
			)
		)
	return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
	global tokenizer
	tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
	examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
	"""
	Converts a list of examples into a list of features that can be directly given as input to a model.
	It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

	Args:
		examples: list of :class:`~transformers.data.processors.squad.SquadExample`
		tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
		max_seq_length: The maximum sequence length of the inputs.
		doc_stride: The stride used when the context is too large and is split across several features.
		max_query_length: The maximum length of the query.
		is_training: whether to create features for model evaluation or model training.
		return_dataset: Default False. Either 'pt' or 'tf'.
			if 'pt': returns a torch.data.TensorDataset,
			if 'tf': returns a tf.data.Dataset
		threads: multiple processing threadsa-smi


	Returns:
		list of :class:`~transformers.data.processors.squad.SquadFeatures`

	Example::

		processor = SquadV2Processor()
		examples = processor.get_dev_examples(data_dir)

		features = squad_convert_examples_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			doc_stride=args.doc_stride,
			max_query_length=args.max_query_length,
			is_training=not evaluate,
		)
	"""

	# Defining helper methods
	features = []
	threads = min(threads, cpu_count())
	with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
		annotate_ = partial(
			squad_convert_example_to_features,
			max_seq_length=max_seq_length,
			doc_stride=doc_stride,
			max_query_length=max_query_length,
			is_training=is_training,
		)
		features = list(
			tqdm(
				p.imap(annotate_, examples, chunksize=32),
				total=len(examples),
				desc="convert squad examples to features",
			)
		)
	new_features = []
	unique_id = 1000000000
	example_index = 0
	for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
		if not example_features:
			continue
		for example_feature in example_features:
			example_feature.example_index = example_index
			example_feature.unique_id = unique_id
			new_features.append(example_feature)
			unique_id += 1
		example_index += 1
	features = new_features
	del new_features
	if return_dataset == "pt":

		# Convert to Tensors and build dataset
		all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
		all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
		all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
		all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
		all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
		all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

		if not is_training:
			all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
			dataset = TensorDataset(
				all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
			)
		else:
			all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
			all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
			dataset = TensorDataset(
				all_input_ids,
				all_attention_masks,
				all_token_type_ids,
				all_start_positions,
				all_end_positions,
				all_cls_index,
				all_p_mask,
				all_is_impossible,
			)

		return features, dataset
	

	return features


class SquadProcessor(DataProcessor):
	"""
	Processor for the SQuAD data set.
	Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
	"""

	def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
		if not evaluate:
			answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
			answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
			answers = []
		else:
			answers = [
				{"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
				for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
			]

			answer = None
			answer_start = None

		return SquadExample(
			qas_id=tensor_dict["id"].numpy().decode("utf-8"),
			question_text=tensor_dict["question"].numpy().decode("utf-8"),
			context_text=tensor_dict["context"].numpy().decode("utf-8"),
			answer_text=answer,
			start_position_character=answer_start,
			title=tensor_dict["title"].numpy().decode("utf-8"),
			answers=answers,
		)

	def get_examples_from_dataset(self, dataset, evaluate=False):
		"""
		Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

		Args:
			dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
			evaluate: boolean specifying if in evaluation mode or in training mode

		Returns:
			List of SquadExample

		Examples::

			import tensorflow_datasets as tfds
			dataset = tfds.load("squad")

			training_examples = get_examples_from_dataset(dataset, evaluate=False)
			evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
		"""

		if evaluate:
			dataset = dataset["validation"]
		else:
			dataset = dataset["train"]

		examples = []
		for tensor_dict in tqdm(dataset):
			examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

		return examples

	def get_train_examples(self, file):
		"""
		Returns the training examples from the data directory.

		Args:
			data_dir: Directory containing the data files used for training and evaluating.
			filename: None by default, specify this if the training file has a different name than the original one
				which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

		"""
		with open(file, "r", encoding="utf-8") as reader:
			input_data = json.load(reader)["data"]
		return self._create_examples(input_data, "train")

	def get_dev_examples(self, file):
		"""
		Returns the evaluation example from the data directory.

		Args:
			data_dir: Directory containing the data files used for training and evaluating.
			filename: None by default, specify this if the evaluation file has a different name than the original one
				which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
		"""
		with open(file, "r", encoding="utf-8") as reader:
			input_data = json.load(reader)["data"]
		return self._create_examples(input_data, "dev")

	def _create_examples(self, input_data, set_type):
		is_training = set_type == "train"
		examples = []
		for entry in tqdm(input_data):
			title = entry["title"] if "title" in entry.keys() else "EMPTY"
			for paragraph in entry["paragraphs"]:
				context_text = paragraph["context"]
				for qa in paragraph["qas"]:
					qas_id = qa["id"]
					question_text = qa["question"]
					start_position_character = None
					answer_text = None
					answers = []

					if "is_impossible" in qa:
						is_impossible = qa["is_impossible"]
					else:
						is_impossible = False

					if not is_impossible:
						if is_training:
							answer = qa["answers"][0]
							answer_text = answer["text"]
							start_position_character = answer["answer_start"]
						else:
							answers = qa["answers"]

					if start_position_character is not None:
						if start_position_character >= len(context_text): continue

					example = SquadExample(
						qas_id=qas_id,
						question_text=question_text,
						context_text=context_text,
						answer_text=answer_text,
						start_position_character=start_position_character,
						title=title,
						is_impossible=is_impossible,
						answers=answers,
					)

					examples.append(example)
		return examples


class SquadExample(object):
	"""
	A single training/test example for the Squad dataset, as loaded from disk.

	Args:
		qas_id: The example's unique identifier
		question_text: The question string
		context_text: The context string
		answer_text: The answer string
		start_position_character: The character position of the start of the answer
		title: The title of the example
		answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
		is_impossible: False by default, set to True if the example has no possible answer.
	"""

	def __init__(
		self,
		qas_id,
		question_text,
		context_text,
		answer_text,
		start_position_character,
		title,
		answers=[],
		is_impossible=False,
	):
		self.qas_id = qas_id
		self.question_text = question_text
		self.context_text = context_text
		self.answer_text = answer_text
		self.title = title
		self.is_impossible = is_impossible
		self.answers = answers

		self.start_position, self.end_position = 0, 0

		doc_tokens = []
		char_to_word_offset = []
		prev_is_whitespace = True


		# Split on whitespace so that different tokens may be attributed to their original position.
		for c in self.context_text:
			if _is_whitespace(c):
				prev_is_whitespace = True
			else:
				if prev_is_whitespace:
					doc_tokens.append(c)
				else:
					doc_tokens[-1] += c
				prev_is_whitespace = False
			char_to_word_offset.append(len(doc_tokens) - 1)

		self.doc_tokens = doc_tokens
		self.char_to_word_offset = char_to_word_offset

# 		print (self.qas_id, start_position_character, len(char_to_word_offset), len(context_text))

		# Start and end positions only has a value during evaluation.
		if start_position_character is not None and not is_impossible:
			self.start_position = char_to_word_offset[start_position_character]
			self.end_position = char_to_word_offset[
				min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
			]


class SquadFeatures(object):
	"""
	Single squad example features to be fed to a model.
	Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
	using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

	Args:
		input_ids: Indices of input sequence tokens in the vocabulary.
		attention_mask: Mask to avoid performing attention on padding token indices.
		token_type_ids: Segment token indices to indicate first and second portions of the inputs.
		cls_index: the index of the CLS token.
		p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
			Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
		example_index: the index of the example
		unique_id: The unique Feature identifier
		paragraph_len: The length of the context
		token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
			If a token does not have their maximum context in this feature object, it means that another feature object
			has more information related to that token and should be prioritized over this feature for that token.
		tokens: list of tokens corresponding to the input ids
		token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
		start_position: start of the answer token index
		end_position: end of the answer token index
	"""

	def __init__(
		self,
		input_ids,
		attention_mask,
		token_type_ids,
		cls_index,
		p_mask,
		example_index,
		unique_id,
		paragraph_len,
		token_is_max_context,
		tokens,
		token_to_orig_map,
		start_position,
		end_position,
		is_impossible,
	):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.token_type_ids = token_type_ids
		self.cls_index = cls_index
		self.p_mask = p_mask

		self.example_index = example_index
		self.unique_id = unique_id
		self.paragraph_len = paragraph_len
		self.token_is_max_context = token_is_max_context
		self.tokens = tokens
		self.token_to_orig_map = token_to_orig_map

		self.start_position = start_position
		self.end_position = end_position
		self.is_impossible = is_impossible


class SquadResult(object):
	"""
	Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

	Args:
		unique_id: The unique identifier corresponding to that example.
		start_logits: The logits corresponding to the start of the answer
		end_logits: The logits corresponding to the end of the answer
	"""

	def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
		self.start_logits = start_logits
		self.end_logits = end_logits
		self.unique_id = unique_id

		if start_top_index:
			self.start_top_index = start_top_index
			self.end_top_index = end_top_index
			self.cls_logits = cls_logits


