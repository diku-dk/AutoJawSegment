import numpy as np
import pandas as pd

a = np.ones((100,2))
a = a + 1.1
a = a.astype(np.float32)

path = '/Users/px/Downloads/view_0.csv'
np.savetxt("foo.csv", a)

image_len = 1000

n_batches = image_len // 30 + 1
ranges = np.array_split(np.arange(0, image_len), n_batches)

for single_range in ranges:
    begin_index = single_range[0]
    end_index = single_range[-1] + 1  #### check if +1 is correct here

    multi_view_preds = np.empty(
        shape=(6, len(single_range)
               , 3),
        dtype=np.float32
    )

    for i in range(6):
        import pandas as pd

        multi_view_preds[i] = pd.read_csv(path, header=None,
                                          nrows=end_index - begin_index,
                                          skiprows=begin_index).to_numpy()

time = pd.read_csv(path, header=None,
                   nrows=110,
                   skiprows=1,
                   #dtype=np.float,
                   ).to_numpy()


time = np.loadtxt(path,
                   max_rows=1000,
                   skiprows=1000,
                   #dtype=np.complex_,
                   ).astype(np.float32)

time
# time = time.to_numpy()
print(time)
print(np.all(time==a))
print(type(time))
print(time.dtype)
