eval "$(conda shell.bash hook)"
echo 'start train teacher'
conda activate py3.10
python3 -u train-teacher.py --config /home/pengyuan/Distilled-DFD-Model/configs/train_teacher.yaml >> train-teacher.log
echo 'complete train teacher'
echo 'start train student'
python3 -u train-student.py --config /home/pengyuan/Distilled-DFD-Model/configs/train_student.yaml >> train-student.log
echo 'complete train student'