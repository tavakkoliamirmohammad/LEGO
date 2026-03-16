from lego import *

i, j, k, w, q, n = symbols('i j k w q n ', integer=True, postive=True)

# add a timing for evalution in ms
import time
start = time.perf_counter()
l = OrderBy(RegP([2, 2, 2, 2, 2], [4, 1, 3, 2, 0])).GroupBy([(2, 2, 2, 2, 2)])
print(l[i, j, k, w, q])
end = time.perf_counter()
print((end - start) * 1000)
