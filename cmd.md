apt-get install libpng-dev libjpeg-dev zlib1g-dev git cmake build-essential pkg-config libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev python3-pip python3-venv python-is-python3 sudo iputils-ping net-tools curl wget libssl-dev libffi-dev python3-setuptools screen tmux htop nvtop zip unzip software-properties-common vim -y

sudo add-apt-repository ppa:deadsnakes/ppa     ppa:fkrull/deadsnakes
sudo apt-get update

export ALL_PROXY="socks5h://10.10.40.206:1082"
export all_proxy="$ALL_PROXY"
export HTTPS_PROXY="$ALL_PROXY"
export https_proxy="$ALL_PROXY"
export HTTP_PROXY="$ALL_PROXY"
export http_proxy="$ALL_PROXY"

unset http_proxy
unset https_proxy


pip install -e .
pip install -r requirements.txt


python -m qwen3_omni_pretrain.cli_train_thinker  --config configs/train/stage1_text_only.yaml  --tokenizer_name_or_path Qwen/Qwen2.5-7B

python -m qwen3_omni_pretrain.cli_train_thinker   --stage stage1   --config configs/train/stage1_text_only.yaml   --tokenizer_name_or_path Qwen/Qwen2.5-7B --tensorboard --log_dir runs/

python -m qwen3_omni_pretrain.cli_train_thinker   --stage stage1   --config configs/train/stage1_text_only.yaml   --tokenizer_name_or_path src/tokenizer/Qwen3/ --tensorboard --log_dir runs/ --resume_from_checkpoint outputs/omni_stage1_text_7b-20251209-003201/step_15000/

torchrun --nproc_per_node=8 -m qwen3_omni_pretrain.cli_train_thinker \
    --stage stage1 \
    --config configs/train/stage1_text_only.yaml \
    --tokenizer_name_or_path Qwen/Qwen2.5-7B

tensorboard --logdir runs/

# quick inference sanity check
python -m qwen3_omni_pretrain.cli_infer_thinker \
    --stage stage1 \
    --checkpoint outputs/qwen3_omni_stage1_text_7b-20251209-003201/best_step_14000 \
    --jsonl data/corpus/val_text.jsonl \
    --max_new_tokens 64

python -m qwen3_omni_pretrain.cli_infer_thinker \
  --stage stage1 \
  --checkpoint outputs/qwen3_omni_stage1_text_7b-20251209-003201/best_step_14000 \
  --chat \
  --max_new_tokens 64

python -m qwen3_omni_pretrain.cli_infer_thinker \
    --stage stage2 \
    --checkpoint outputs/20251208-184741/best_step_10000 \
    --jsonl data/corpus/stage2_omni_val.jsonl \
    --image_root data \
    --audio_root data \
    --max_new_tokens 64

python -m qwen3_omni_pretrain.cli_infer_thinker \
  --stage stage2 \
  --checkpoint outputs/20251208-184741/best_step_10000 \
  --chat \
  --max_new_tokens 64

git config --global user.email eliot.zhao@finnox.cn
git config --global user.name "Eliot Zhao"

netstat -tulnp | grep 29500