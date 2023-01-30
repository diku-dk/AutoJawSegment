import os

import numpy as np
import tensorflow as tf

import time
class father():
    def __init__(self):
        self.a = 0
        setattr(eval('self'), 'a', 1)
        setattr(self, 'ab', 1)

    @property
    def dim(self):
        print(getattr(self, 'a'))
        return self.a



class child(father):
    def __init__(self):
        super().__init__()

    @property
    def dim(self):
        return self.a + 1

for i, j in zip(np.ones((3,3)), np.ones((3,3))):
    print(i)

def compare(a):

    for _ in range(20):
        a = np.random.randint(0, 2, 10**8)
        b = np.random.randint(0, 2, 10**8)
        start = time.time()
        np.sum(np.logical_xor(a, b))
        print('np xor takes', time.time() - start)

        start = time.time()
        np.sum(a ^ b)
        print('builtin xor takes', time.time() - start)

        start = time.time()
        np.sum(np.abs(a - b))
        print('abs minus takes', time.time() - start)

        start = time.time()
        np.linalg.norm((a-b), ord=1)
        print('abs minus takes', time.time() - start)

class gen():
    def __int__(self):
        pass

    def __getitem__(self, item):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        w = np.random.randn(3, 3)
        return x, y, w

    def __len__(self):
        """ Undefined, return some high number - a number is need in keras """
        return int(10**12)

    def __call__(self):
        def tensor_iter():
            """ Iterates the dataset, converting numpy arrays to tensors """
            for x, y, w in self:
                yield (tf.convert_to_tensor(x),
                       tf.convert_to_tensor(y),
                       tf.convert_to_tensor(w))
        return tensor_iter()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

if __name__ == '__main__':
    sb = 2

    print(os.getcwd())
    print('ok')

    a = father()
    print(a.dim)


    import time
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    pool = ThreadPoolExecutor(max_workers=7)
    # pool = ProcessPoolExecutor(max_workers=7)

    offsets = np.arange(1, 1e5, 5)

    def _do(offset, ind):
        offset += 1
        return offset
    inds = np.arange(offsets.shape[0])

    begin = time.time()
    result = pool.map(_do, offsets, inds)
    print(f'time with multithread = {time.time()-begin}')

    #pool.shutdown()

    num_vals = 0
    for i in enumerate(result):
        num_vals += 1
    print(num_vals)


    with ThreadPoolExecutor() as p:
        processing = p.map(_do, offsets, inds)
        for process in processing:
            print(process)

    with ProcessPoolExecutor() as p:
        processing = p.map(_do, offsets, inds)
        for process in processing:
            print(process)

    begin = time.time()
    result = []
    for i, j in zip(offsets, inds):
        result.append(_do(i, j))
    print(f'time without multithread = {time.time()-begin}')

    begin = time.time()
    result = np.zeros(len(offsets))
    for i, j in zip(offsets, inds):
        result[j] = _do(i, j)
    print(f'time without multithread = {time.time()-begin}')


    breakpoint()




    # !import code; code.interact(local=vars())
    # interact
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train = gen()
    dtypes, shapes = list(zip(*map(lambda x: (x.dtype, x.shape), train[0])))
    train = tf.data.Dataset.from_generator(train, dtypes, shapes)
    train = train.with_options(options)
    a = child()
    a.dim

    hd = 66
    j = complex(512)
    offset_from_center = -100
    grid = np.mgrid[-hd:hd:j,
                    -hd:hd:j,
                    offset_from_center:offset_from_center:1j]

