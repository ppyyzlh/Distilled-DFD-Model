eval "$(conda shell.bash hook)"
echo 'start train teacher'
conda activate py3.10
python3 -u train-teacher.py --config configs/train_teacher.yaml
echo 'complete train teacher'
echo 'start train student'
python3 -u train-student.py --config configs/train_student.yaml
echo 'complete train student'