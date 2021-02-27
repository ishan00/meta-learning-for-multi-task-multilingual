## Meta-Learning for Effective Multi-task and Multilingual Modelling
Link to the paper - https://arxiv.org/abs/2101.10368

### Abstract
```
Natural language processing (NLP) tasks (e.g. question-answering in English) benefit from knowledge of other tasks
(e.g. named entity recognition in English) and knowledge of other languages (e.g. question-answering in Spanish). Such
shared representations are typically learned in isolation, either across tasks or across languages. In this work, we 
propose a meta-learning approach to learn the interactions between both tasks and languages. We also investigate the
role of different sampling strategies used during meta-learning. We present experiments on five different tasks and
six different languages from the XTREME multilingual benchmark dataset. Our meta-learned model clearly improves in
performance compared to competitive baseline models that also include multi-task baselines. We also present zero-shot
evaluations on unseen target languages to demonstrate the utility of our proposed model.
```

### Introduction



### Dataset and Directory Structure

We use the recently released [XTREME](https://github.com/google-research/xtreme) benchmark for dataset. We choose a subset of 5 tasks - Question Answering, Natural Language Answering, Paraphrase Identification, Part-of-Speech tagging and Named Entity Recognition and 6 languages - English, Hindi, French, Chinese, Spanish and German for our experiments. Later on for zero-shot result we use many other languages from these tasks. The authors of XTREME were kind enough to provide us with the in-house translated-train datasets whenever not publically available.

`datapath.py` lists all dataset files used in the paper. The naming of datasets is done using XX_YY where XX is the task - Question Answering (qa), Natural language inference (sc), Paraphrase identification (pa), Part-of-speech tagging (po) and Named entity recognition (tc), and YY is the language - English (en), Hindi (hi),... etc

### Running Meta-Learning Experiments

#### Temperature Sampling
#### DDS Sampling

### Other Experiments
#### Multi-task 
#### Zero-shot Experiments

We evaluate trained meta models in zero-shot setting on multiple languages for each task. The list of zero-shot languages for each task can be found in `datapath.py` file under the key `test`. The below command evaluates the model on NER for Tamil

```
python3 zeroshot.py --load PATH/TO/MODEL --task tc_ta
```

--------
#### Contact
Ishan Tarunesh - ishantarunesh@gmail.com <br>
Sushil Khyalia - skhyalia2014@gmail.com
