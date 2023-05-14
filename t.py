import numpy as np
my_list = []
a, b, c = np.zeros((1, 2, 3)), np.zeros((3, 2, 3)), np.zeros((1, 2, 3))
my_list.append(a)
my_list.append(b)
my_list.append(c)
print(my_list)
result = np.concatenate(my_list, axis=0)

print(result.shape)