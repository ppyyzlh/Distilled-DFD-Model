import os
# 判断目录是否可读
dir_path = "/some/path/to/dir"
dir_readable = os.access('minivideos/mini_train_vedios_1/aagfhgtpmv.mp4', os.R_OK)
print(dir_readable) # True or False