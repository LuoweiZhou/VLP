# VLP
This repo hosts the source code for our AAAI2020 work [Vision-Language Pre-training (VLP)](https://arxiv.org/pdf/1909.11059.pdf).
We have released the pre-trained model on [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) dataset
and fine-tuned models on COCO Captions and Flickr30k for image captioning and VQA 2.0 for VQA.


## Installation
### Conda Environment (Option I, Recommended)
0) Recursively ssh clone the repo to include `coco` and `pythia` submodules.
```
git clone --recursive git@github.com:LuoweiZhou/VLP.git
```
or clone with https:
```
git clone --recursive https://github.com/LuoweiZhou/VLP.git
```

1) Install CUDA (e.g., 10.0), CUDNN (e.g., v7.5), and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (either Miniconda2 or 3, version 4.6+).

2) Run the following commands to set up conda env and install Python packages:

```
MINICONDA_ROOT=[to your Miniconda root directory] # e.g., /home/[usrname]/miniconda3
cd VLP
conda env create -f misc/vlp.yml --prefix $MINICONDA_ROOT/envs/vlp
conda activate vlp
```

3) Finally, `cd` to the repo root directory and install other dependencies by running:
```
./setup.sh
```
To support language evaluation (SPICE), run
```
cd coco-caption
./get_stanford_models.sh
```

### Docker Image (Option II)
First, install or upgrade to the latest [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (e.g., set `<VERSION_STRING>` to `5:19.03.2~3-0~ubuntu-xenial`). Then pull our docker image:
```
docker pull luzhou/vlp
```

Before running the container, you need to declare the environment variable to your data root (`$DATA_ROOT`, see [data prep](#data_prep)) and it will be attached as a volume to our container. Finally, install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster) and run the docker image in a fresh container:
```
docker run --gpus all --name vlp_container -it \
     -v $DATA_ROOT:/mnt/dat \
     --shm-size 8G -p 8888:8888 vlp /bin/bash
```

You can know more about docker commands and usages [here](https://docs.docker.com/engine/reference/commandline/docker/).

(Optional) To build the image on your own,
```
docker build -t vlp .
```


## <a name='data_prep'></a> Data Preparation
Download links for dataset annotations and features: COCO Captions+VQA 2.0 ([Part I(95GB)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212019&authkey=ACn4bwZ0nmZ0nik), [Part II(79GB)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212018&authkey=AHoTGG-7-6kwoAY), download both and run `cat COCO0* > COCO.tar.gz`), [Flickr30k Captions(27GB)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212015&authkey=AFZ2iehPM8HREeA). **If you prefer to download with `wget`, we attach the commands [here](#misc)**.
Then, uncompress the downloaded files and place under your data root (denoted as `DATA_ROOT`).

To prepare for the pre-training, first download and uncompress our pre-processed Conceptual Captions (CC) [data(6GB)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%213781&authkey=ANA--esfJnWIKIE) and place under your data root. Then, download and uncompress the region features from Google Drive ([feat(509GB)](https://drive.google.com/file/d/14mr49-14-ZjJXOohInzoOLBZlJb_y7fh/view?usp=sharing), [cls(468GB)](https://drive.google.com/file/d/1kRlnQJcTjGFaOHSptekgG98MiCsTQYDt/view?usp=sharing)) under the `CC/region_feat_gvd_wo_bgd/feat_cls_1000_float16` dir.  To evaluate CC on caption generation, download the reference [file](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212017&authkey=AHy5eiJM75RwPxg) and place it under `coco-caption/annotations`.

Besides, download and uncompress the detectron fc7 weight files under the code root directory (denoted as `CODE_ROOT`): [GVD Detectron fc7](https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz).

(Optional, only for VQA) Download the VQA 2.0 annotation (based on [Pythia](https://github.com/facebookresearch/pythia#data)):
```
cd $CODE_ROOT/pythia
mkdir -p data && cd data
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz
tar xf vocab.tar.gz && rm vocab.tar.gz

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip && rm v2_Annotations_Val_mscoco.zip

mkdir -p imdb && cd imdb
wget https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz
tar xf vqa.tar.gz && rm vqa.tar.gz
```

(Optional, only for pre-training) Download the [UniLM](https://arxiv.org/abs/1905.03197) [checkpoints](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212016&authkey=AB5-lxzCkgpfLhg) and uncompress under your checkpoint root (denoted as `CHECKPOINT_ROOT`).


## Experiment Overview
**Most of the experiments in this work are performed on 8x V100 GPUs with distributed data parallel (i.e., set `--world_size` to 8, `--local_rank` and `--global_rank` from 0 to 7 with 8 separate scripts), unless specified otherwise.** See below for detailed configurations (also in the Appendix of the paper).

| Dataset | Batch Size | Learning Rate | # of Epochs | GPUs | Time per Epoch |
| ----- | -----:| -----:| -----:| -----:| -----:|
| CC | 64(x8) | 1e-4(x8) | 30 | 8x V100 | 5hr |
| COCO | 64(x8) | 3e-5(x8) | 30 | 8x V100 | 12min |
| VQA 2.0 | 64(x2) | 2e-5(x2) | 20 | 2x V100 | 32min |
| Flickr30k | 64(x8) | 3e-5(x8) | 30 | 8x V100 | 3min |
| COCO (w/o pre-training) | 64(x8) | 3e-4(x8) | 30 | 8x V100 | 12min |
| COCO (SCST training) | 16(x4) | 1e-6(x4) | 30 | 4x Titan Xp | 3hr |

The `(x2), (x4), (x8)` in the batch size and learning rate results from distributed data parallel. Gradients are accumulated/added across GPUs.

**Note that some modules need to be imported manually:**

```
export PYTHONPATH=$CODE_ROOT/pythia:$CODE_ROOT/pythia/pythia/legacy:$CODE_ROOT:$PYTHONPATH
```


## Pre-training
An example code on single-GPU training:
```
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_cc} \
    --model_recover_path $CHECKPOINT_ROOT/bert_save/base_model_pretrained/model_153999_cpu.bin \
    --do_train --learning_rate ${lr} --new_segment_ids --always_truncate_tail --amp \
    --src_file $DATA_ROOT/CC/annotations/dataset_cc.json \
    --dataset cc --split train --file_valid_jpgs $DATA_ROOT/CC/annotations/cc_valid_jpgs.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob ${w_s} --bi_prob ${w_b} --image_root $DATA_ROOT/CC/region_feat_gvd_wo_bgd \
    --region_bbox_file bbox/cc_detection_vg_thresh0.2_feat_gvd_checkpoint_trainval.h5 \
    --region_det_file_prefix feat_cls_1000_float16/cc_detection_vg_100dets_gvd_checkpoint_trainval
```
where `lr=1e-4`, `w_s=0.75`, `w_b=0.25`, and `checkpoint_cc` is the id of the checkpoint. The pre-trained models are available [here](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212026&authkey=AH98pIVaNS4apSI).


## Fine-tuning
The fine-tuning checkpoints are available at: [COCO (CE optim)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212028&authkey=AEjQxFF1FcBK-Aw), [COCO (CIDEr optim)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0), [VQA 2.0 (train on train set only)](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212029&authkey=APjfGJd1-nzDO7s), [Flickr30k](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212030&authkey=AGmfQ0fXcYCQun0).

### COCO Captions
An example code on single-GPU training:
```
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_coco_ce} \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_cc}/model.30.bin \
    --do_train --new_segment_ids --always_truncate_tail --amp \
    --src_file $DATA_ROOT/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json \
    --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd --enable_butd --s2s_prob 1 --bi_prob 0
```

(Optional) To enable Self-Critical Sequence Training (SCST), set `--model_recover_path $CHECKPOINT_ROOT/${checkpoint_coco_ce}/model.28.bin`, `--max_pred 0`, `--mask_prob 0`, `--scst`, `--learning_rate 1e-6` (note that SCST requires a much smaller lr than the default `3e-5`), and `--output_dir` accordingly. The training takes 30 epochs to converge with each epoch takes roughly 3hr.

An example code on 2-GPU training with distributed data parallel:
```
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_coco_ce} \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_cc}/model.30.bin \
    --do_train --new_segment_ids --always_truncate_tail --amp \
    --src_file $DATA_ROOT/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json \
    --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd --enable_butd --s2s_prob 1 --bi_prob 0 \
    --local_rank 0 --global_rank 0 --world_size 2 &
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_coco_ce} \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_cc}/model.30.bin \
    --do_train --new_segment_ids --always_truncate_tail --amp \
    --src_file $DATA_ROOT/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json \
    --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd --enable_butd --s2s_prob 1 --bi_prob 0 \
    --local_rank 1 --global_rank 1 --world_size 2
```


### VQA 2.0
An example code on single-GPU training:
```
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_vqa2} \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_cc}/model.30.bin \
    --do_train --learning_rate 2e-5 --new_segment_ids --always_truncate_tail --amp \
    --num_train_epochs 20 --enable_butd --s2s_prob 0 --bi_prob 1 \
    --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd
    --tasks vqa2 --src_file $CODE_ROOT/pythia/data/imdb/vqa/imdb_train2014.npy \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json \
    --mask_prob 0 --max_pred 1
```

To get the models for leaderboard, we perform the training on both train set and val set (set `src_file` to `imdb_train2014` and `imdb_val2014`).

### Flickr30k Captions
```
python vlp/run_img2txt_dist.py --output_dir $CHECKPOINT_ROOT/${checkpoint_flickr30k} \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_cc}/model.30.bin \
    --do_train --new_segment_ids --always_truncate_tail --amp \
    --image_root $DATA_ROOT/flickr30k/region_feat_gvd_wo_bgd --enable_butd --s2s_prob 1 --bi_prob 0 \
    --dataset flickr30k --region_bbox_file $DATA_ROOT/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 \
    --src_file $DATA_ROOT/flickr30k/annotations/dataset_flickr30k.json \
    --file_valid_jpgs $DATA_ROOT/flickr30k/annotations/flickr30k_valid_jpgs.json
```


## Inference and Testing
Here, we list the expected result outcomes from our Unified VLP checkpoints.
For image captioning, on Karpathy's test split:

| Dataset | Method | BLEU@4 | METEOR | CIDEr | SPICE |
| ----- | -----:| -----:| -----:| -----:| -----:|
| COCO | Unified VLP | 36.5 | 28.4 | 116.9 | 21.2 |
|  | Unified VLP + SCST | 39.5 | 29.3 | 129.3 | 23.2 |
| Flickr30k | Unified VLP | 30.1 | 23.0 | 67.4 | 17.0 |

For VQA:

| Dataset | Trained on | Eval Split | Overall | Yes/No | Number | Other |
| ----- | -----:| -----:| -----:|-----:| -----:| -----:|
| VQA 2.0 | train only | Dev | 67.4 | 85.4 | 50.1 | 58.3 |
| | train+val | Test-Dev | 70.5 | 87.2 | 52.1 | 60.3 |
| | train+val | Test-Standard | 70.7 | 87.4 | 52.1 | 60.5 |

**Note that results on Test-Dev and Test-Standard are from VQA 2.0 evaluation [server](https://evalai.cloudcv.org/web/challenges/challenge-page/163/submission). `train+val` indicates models are trained on both training set and validation set following the practice from early works.**

Note: All the evaluation scripts support data parallel. But since we do not use standard PyTorch DataLoader, the data loading speed might be the bottleneck (imagine `num_workers` is always 0). We recommend to perform single-GPU inference (e.g., `CUDA_VISIBLE_DEVICES=0`).

### COCO Captions
```
python vlp/decode_img2txt.py \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_coco_ce}/model.${epoch}.bin \
    --new_segment_ids --batch_size 100 --beam_size ${beam} --enable_butd \
    --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd/ --split ${split} \
    --src_file $DATA_ROOT/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json
```
where `checkpoint_coco_ce` indicates checkpoint name, `beam=1` for `split=val` set and `5` for `split=test` set, and `epoch` indicates the checkpoint at which epoch.

### VQA 2.0
```
python vlp/eval_vqa2.py \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_vqa2}/model.${epoch}.bin \
    --new_segment_ids --enable_butd --image_root $DATA_ROOT/COCO/region_feat_gvd_wo_bgd/ \
    --src_file $CODE_ROOT/pythia/data/imdb/vqa/imdb_${split}.npy --batch_size 50 \
    --file_valid_jpgs $DATA_ROOT/COCO/annotations/coco_valid_jpgs.json --split ${split}
```
where `split` could be `val2014` or `test2015`.

### Flickr30k Captions
```
python vlp/decode_img2txt.py \
    --model_recover_path $CHECKPOINT_ROOT/${checkpoint_flickr30k}/model.${epoch}.bin \
    --new_segment_ids --batch_size 100 --beam_size ${beam} --enable_butd \
    --image_root $DATA_ROOT/flickr30k/region_feat_gvd_wo_bgd/ --split ${split} \
    --dataset flickr30k --region_bbox_file $DATA_ROOT/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 \
    --src_file $DATA_ROOT/flickr30k/annotations/dataset_flickr30k.json \
    --file_valid_jpgs $DATA_ROOT/flickr30k/annotations/flickr30k_valid_jpgs.json
```
where `beam=1` for `split=val` set and `5` for `split=test` set, and `epoch` indicates the checkpoint at which epoch.

### Testing
For all the datasets, checkpoints (by epochs) with the best validation accuracy (CIDEr in captioning and overall accuracy in VQA) are evaluated on the test set (Test-Dev and Test-Standard for VQA 2.0).


## <a name='misc'></a> Misc
The Detectron-based feature extraction code is available under this [repo](https://github.com/LuoweiZhou/detectron-vlp). You need to download this [config](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212013&authkey=AHIvnE1FcggwiLU) file and [checkpoint](https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU) file.

List of download commands (only for OneDrive):
```
wget -O caption_cc_val.json "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212017&authkey=AHy5eiJM75RwPxg"

# data
wget -O COCO00 "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212019&authkey=ACn4bwZ0nmZ0nik"
wget -O COCO01 "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212018&authkey=AHoTGG-7-6kwoAY"
wget -O flickr30k.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212015&authkey=AFZ2iehPM8HREeA"
wget -O CC.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%213781&authkey=ANA--esfJnWIKIE"

# UniLM checkpoint
wget -O bert_save.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212016&authkey=AB5-lxzCkgpfLhg"

# pre-training checkpoints
wget -O cc_g8_lr1e-4_batch512_s0.75_b0.25.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212026&authkey=AH98pIVaNS4apSI"

# fine-tuning checkpoints
wget -O coco_g8_lr3e-5_batch512_ft_from_s0.75_b0.25.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212028&authkey=AEjQxFF1FcBK-Aw"
wget -O coco_g4_lr1e-6_batch64_scst.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0"
wget -O vqa2_g2_lr2e-5_batch512_ft_from_s0.75_b0.25.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212029&authkey=APjfGJd1-nzDO7s"
wget -O flickr30k_g8_lr3e-5_batch512_ft_from_s0.75_b0.25.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212030&authkey=AGmfQ0fXcYCQun0"

# Detectron config/model
wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.yaml "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212013&authkey=AHIvnE1FcggwiLU"
wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
```


## Reference
Please acknowledge the following paper if you use the code:
```
@article{zhou2019vlp,
  title={Unified Vision-Language Pre-Training for Image Captioning and VQA},
  author={Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao},
  journal={arXiv preprint arXiv:1909.11059},
  year={2019}
}
```


## Related Projects/Codebase
- Pre-trained UniLM: https://github.com/microsoft/unilm
- GVD (captioing+grounding): https://github.com/facebookresearch/grounded-video-description
- Video DenseCap: https://github.com/salesforce/densecap
- MT-DNN: https://github.com/namisan/mt-dnn


## Acknowledgement
Our code is mainly based on [Li Dong](http://homepages.inf.ed.ac.uk/s1478528/) et al.'s [UniLM](https://github.com/microsoft/unilm) repo. Also, a part of the code is based on [pytorch-transformers v0.4.0](https://github.com/huggingface/pytorch-transformers/tree/v0.4.0) and [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). We thank the authors for their wonderful open-source efforts.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [UniLM](https://github.com/microsoft/unilm/blob/master/LICENSE) project and [pytorch-transformers v0.4.0](https://github.com/huggingface/transformers/blob/master/LICENSE) project.
