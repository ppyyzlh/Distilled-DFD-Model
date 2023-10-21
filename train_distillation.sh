eval "$(conda shell.bash hook)"
echo 'start train teacher'
conda activate py3.10
python3 -u train-distillation.py --config configs/train_distillation.yaml
echo 'complete train'