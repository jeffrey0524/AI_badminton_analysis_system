import dataprocess
data = dataprocess.get_data_list('../test_part1/train/')
list = []
for aa in data:
    list.append(aa[2])
print(list)