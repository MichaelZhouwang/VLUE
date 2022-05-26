# VLUE
This repo contains data, descriptions, and codes for baselines in the Visual Language Understanding Evaluation (VLUE) benchmark. See our paper for more details about VLUE or the baselines.

## Tasks

VLUE considers 5 representative vision-and-language tasks including

## Downloading VLUE

### In-Domain Data 

You can download data for [download json files](https://drive.google.com/file/d/1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4/view?usp=sharing) 



## Evaluation on VLUE
We provide the examples to run VLUE test on [X-VLM](https://github.com/zengyan-97/X-VLM) as follows: 
```angular2html
cp -r data/ X-VLM/data/vlue_released
cd X-VLM

python3 run.py --task "eval_vlue_itr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "itr_coco/checkpoint_9.pth"

python3 run.py --task "eval_vlue_vqa" --dist "1" --evaluate  --output_dir "output/" --checkpoint "vqa/model_state_epoch_9.th"

python3 run.py --task "eval_vlue_nlvr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "nlvr/nlvr_ft/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco_bbox/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco_weakly" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco/checkpoint_best.pth"
```
