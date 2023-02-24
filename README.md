# Learn What Is Possible, Then Choose What Is Best: Disentangling One-To-Many Relations in Language Through Text-based Games

This repository contains the source code for our EMNLP-FINDINGS 2022 paper [Learn What Is Possible, Then Choose What Is Best: Disentangling One-To-Many Relations in Language Through Text-based Games](https://aclanthology.org/2022.findings-emnlp.364/).

# Overview

Language models pre-trained on large self-supervised corpora, followed by task-specific fine-tuning has become the dominant paradigm in NLP. These pre-training datasets often have a one-to-many structure—e.g. in dialogue there are many valid responses for a given context. However, only some of these responses will be desirable in our downstream task. This raises the question of how we should train the model such that it can emulate the desirable behaviours, but not the undesirable ones. Current approaches train in a one-to-one setup—only a single target response is given for a single dialogue context—leading to models only learning to predict the average response, while ignoring the full range of possible responses. Using text-based games as a testbed, our approach, PASA, uses discrete latent variables to capture the range of different behaviours represented in our larger pre-training dataset. We then use knowledge distillation to distil the posterior probability distribution into a student model. This probability distribution is far richer than learning from only the hard targets of the dataset, and thus allows the student model to benefit from the richer range of actions the teacher model has learned. Results show up to 49% empirical improvement over the previous state-of-the-art model on the Jericho Walkthroughs dataset.

# Getting Started

First, you will want to download the ClubFloyd dataset from: https://github.com/princeton-nlp/calm-textgame/tree/master/calm. The paper also makes use of the Jericho Walkthroughs, the preprocessed versions of which are included in this repo. The original repo for the Jericho framework can be found at: https://github.com/microsoft/jericho.

The requirements for the code are:

- transformers
- pytorch
- datasets
- spacy
- nltk
- scikit-learn
- scipy

The entry point for training the model is `train.py`. 

(1) we pre-train the teacher model on ClubFloyd with:

```![pasa_overview](https://user-images.githubusercontent.com/71493502/221183552-2ebb05a1-0665-4b22-aa9e-bd262f9c46ad.png)

python train.py --model_type latent \
  --task_type clubfloyd \
  --data_directory $PATH/TO/CLUBFLOYD$ \
  --intent_type regex \
  --model_path distilbert-base-uncased \
  --output_dir clubfloyd_regex \
  --epochs 3
```
(2) We then similarly pre-train the student model on ClubFloyd with:
```
python train.py --model_type baseline \
  --task_type clubfloyd \
  --data_directory $PATH/TO/CLUBFLOYD$ \
  --model_path distilbert-base-uncased \
  --output_dir clubfloyd_baseline \
  --epochs 3
 ```
 (3) Fine-tune the teacher model on Jericho Walkthroughs:
 ```
python train.py --model_type latent \
  --task_type jericho \
  --intent_type regex \
  --model_path clubfloyd_regex \
  --output_dir jericho_regex \
  --epochs 1
 ```
 (4) Knowledge distillation on Jericho Walkthroughs:
 ```
 python train.py --model_type distillation \
  --task_type jericho \
  --intent_type regex \
  --student_path clubfloyd_baseline \
  --teacher_path jericho_regex \
  --output_dir distillation \
  --output_dir jericho_regex \
  --epochs 1
 ```
 (5) Obtain game-by-game evaluation results on the Jericho test data:
 ```
 python eval.py --model_path distillation \
  --model_type baseline \
  --task_type jericho \
 ```
 
 # Citation
 Please cite our paper if you found PASA useful in your work:
 ```bibtex
 @inproceedings{towle-zhou-2022-learn,
    title = "Learn What Is Possible, Then Choose What Is Best: Disentangling One-To-Many Relations in Language Through Text-based Games",
    author = "Towle, Benjamin  and
      Zhou, Ke",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.364",
    pages = "4955--4965"
}
```
 
