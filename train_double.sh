eval "$(conda shell.bash hook)"
echo 'start train teacher'
conda activate ppyy3.7
python3 -u train-teacher.py >> train-teacher.log
echo 'complete train teacher'
echo 'start train student'
python3 -u train-student.py >> train-student.log
echo 'complete train student'

