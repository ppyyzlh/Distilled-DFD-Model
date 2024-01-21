eval "$(conda shell.bash hook)"
echo 'start train teacher'
conda activate py3.10
python3 -u train_distillation.py --config configs/train_distillation_single.yaml
echo 'complete train'