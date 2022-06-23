# PASA

This is the source code corresponding to the EMNLP 22 submission: Learn What Is \textit{Possible}, Then Choose What Is \textit{Best}: Disentangling One-To-Many Relations in Language Through Text-based Games.

First, you will want to download the ClubFloyd dataset from: https://github.com/princeton-nlp/calm-textgame/tree/master/calm.

The entry point for training the model is `train.py`. 

(1) we pre-train the teacher model on ClubFloyd with:

```
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
 
 
