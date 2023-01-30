import argparse
import os

import tensorflow as tf

current_path = os.path.abspath(__file__)
home_dir = os.path.dirname(os.path.dirname(current_path))

def get_argparser():
    parser = argparse.ArgumentParser(usage='this is just a parser')
    parser.add_argument("args", help="Arguments passed to script",
                        nargs=argparse.REMAINDER)
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch_size")
    parser.add_argument("--memory_size", type=int, default=1024,
                        help="memory_size")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--model", type=str, default="VAE")


    return parser


def get_kwargs(args=None):

    parser = get_argparser()
    parsed = parser.parse_args(args)

    kwargs = {}
    kw_test_methods = {}
    kwargs['set_memory'] = parsed.limit_memory
    kwargs['batch_size'] = parsed.batch_size
    kwargs['kl_weight'] = parsed.kl_weight
    kwargs['model_name'] = parsed.model if parsed.model_name == '' else parsed.model_name

    return kwargs



def entry_func(args=None):
    kwargs, model, data_gen, kw_test_methods = get_kwargs(args)

    print(kwargs)
    batch_size = kwargs['batch_size']

    # data_gen = mnist_vae_loader(batch_size=batch_size, flatten=False)

    train_dataset, test_dataset = data_gen.get_dataset()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if kwargs['with_label'] or 'grad' in kwargs['model_name'] and 'VAE_grad_loss' not in kwargs['model_name']:
        train_batch_x, train_batch_y = iter(train_dataset).next()
        test_batch_x, test_batch_y = iter(test_dataset).next()
    else:
        train_batch_x = iter(train_dataset).next()
        test_batch_x = iter(test_dataset).next()
        train_batch_y = None
        test_batch_y = None

    test = tester(model=model, train_batch=train_batch_x, val_batch=test_batch_x,
                  train_label=train_batch_y, val_label=test_batch_y,
                  **kwargs
                  )
    test.restore_model()

    for method, v in kw_test_methods.items():
        if v:
            getattr(test, method)()

    mean, var = test.get_latent_mean_var()
    print(f'mean {mean}')
    print(f'var {var}')
    #
    # test.resample_from_latent_given_range()
    #
    # # test.draw_only_final()
    # #
    # test.draw_reonstruct()
    #
    #
    # test.eval_numerically()
    #
    # test.resample_from_latent(batch_size=32)
    #
    # test.plot_label_clusters()

if __name__ == "__main__":
    # sb = '--num_GPUs 3 --num_GPUs=3 --no_argmax'

    entry_func()