import os, create_dataset

file = open('log-train', 'r')
lines = list(map(lambda x: x.strip().split(' ')[1], file.readlines()))
print(lines)

for path, dir_list, file_list in os.walk('alldata'):
    file_list = list(map(lambda x: os.path.join(path, x), file_list))
    file_list.sort()
    print(file_list)
    create_dataset.createDataset('trainset_lmdb', file_list, lines, None, True)
    