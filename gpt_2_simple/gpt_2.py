import tarfile
import os
import json
import requests
import sys
import shutil
import re
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
import time
from datetime import datetime
import csv
import argparse

# if in Google Colaboratory
try:
    from google.colab import drive
except:
    pass

from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer

assert tf.__version__ < '2.0.0', "gpt-2-simple currently does not support " \
    "TensorFlow 2.0. You'll need to use a virtualenv/cloud computer which " \
    "has Tensorflow 1.X on it."


def download_file_with_progress(url_base, sub_dir, model_name, file_name):
    """General utility for incrementally downloading files from the internet
    with progress bar
    from url_base / sub_dir / filename
    to local file system sub_dir / filename

    Parameters
    ----------
    file_name : str
        name of file to get e.g. "hparams.json"
    sub_dir: str
        subdirectory inside which to get and copy locally eg. "models/124M" 
        no trailing slash
    url_base : str
        Start of URL location specifying server and any base directories no 
        trailing slash
        e.g. "https://storage.googleapis.com/gpt-2"
    """

    # set to download 1MB at a time. This could be much larger with no issue
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    r = requests.get(url_base + "/models/" + model_name + "/" + file_name, stream=True)
    with open(os.path.join(sub_dir, file_name), 'wb') as f:
        file_size = int(r.headers["content-length"])
        with tqdm(ncols=100, desc="Fetching " + file_name,
                  total=file_size, unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)
   

def download_gpt2(model_dir='models', model_name='124M'):
    """Downloads the GPT-2 model into the current directory
    from Google Cloud Storage.

    Parameters
    ----------
    model_dir : str
        parent directory of model to download

    model_name : str
        name of the GPT-2 model to download. 
        As of 22 May 2019 one of "124M" or "355M" but may later include other 
        model sizes

    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    """

    # create the <model_dir>/<model_name> subdirectory if not present
    sub_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_dir = sub_dir.replace('\\', '/')  # needed for Windows

    for file_name in ['checkpoint', 'encoder.json', 'hparams.json',
                      'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                      'model.ckpt.meta', 'vocab.bpe']:
        download_file_with_progress(url_base="https://storage.googleapis.com/gpt-2",
                                    sub_dir=sub_dir,
                                    model_name=model_name,
                                    file_name=file_name)


def start_tf_sess(threads=-1, server=None):
    """
    Returns a tf.Session w/ config
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    if server is not None:
        return tf.compat.v1.Session(target=server.target, config=config)
    
    return tf.compat.v1.Session(config=config)


def reset_session(sess, threads=-1, server=None):
    """Resets the current TensorFlow session, to clear memory
    or load another model.
    """

    tf.compat.v1.reset_default_graph()
    sess.close()
    sess = start_tf_sess(threads, server)
    return sess

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def finetune(sess,
             dataset,
             steps=-1,
             model_name='124M',
             model_dir='models',
             combine=50000,
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             restore_from='latest',
             run_name='run1',
             checkpoint_dir='checkpoint',
             sample_every=100,
             sample_length=1023,
             sample_num=1,
             multi_gpu=False,
             save_every=1000,
             print_every=1,
             max_checkpoints=1,
             use_memory_saving_gradients=False,
             only_train_transformer_layers=False,
             optimizer='adam',
             overwrite=False):
    """Finetunes the model on the given dataset.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    # assert model_name not in ['774M', '1558M'] or multi_gpu, "Currently, a modern single GPU cannot finetune the 774M GPT-2 model or larger."

    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(checkpoint_dir, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        try:
            shutil.copyfile(os.path.join(model_dir, model_name, file),
                            os.path.join(checkpoint_path, file))
        except FileNotFoundError as fnf_error:
            print("You need to download the GPT-2 model first via download_gpt2()")
            raise(fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if model_name not in ['117M', '124M']:
        use_memory_saving_gradients = True
        only_train_transformer_layers = True
        accumulate_gradients = 1

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    gpus = []

    if multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus)
    loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=sample_length,
        context=context,
        batch_size=batch_size,
        temperature=1.0,
        top_k=40)

    all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars

    if optimizer == 'adam':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    if accumulate_gradients > 1:
        if use_memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)
    else:
        if use_memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(ys=loss, xs=train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.compat.v1.summary.scalar('loss', loss)

    summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)

    saver = tf.compat.v1.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.compat.v1.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join(model_dir, model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join(model_dir, model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter-1))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter-1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter-1) + '\n')

    def generate_samples():
        context_tokens = data_sampler.sample(1)
        all_text = []
        index = 0
        while index < sample_num:
            out = sess.run(
                tf_sample,
                feed_dict={context: batch_size * [context_tokens]})
            for i in range(min(sample_num - index, batch_size)):
                text = enc.decode(out[i])
                text = '======== SAMPLE {} ========\n{}\n'.format(
                    index + 1, text)
                all_text.append(text)
                index += 1
        print(text)
        maketree(os.path.join(SAMPLE_DIR, run_name))
        with open(
                os.path.join(SAMPLE_DIR, run_name,
                             'samples-{}').format(counter), 'w') as fp:
            fp.write('\n'.join(all_text))

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(batch_size)]

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))
        save()

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    if steps:
        steps = int(steps)
    
    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                save()
                return
            if (counter - 1) % save_every == 0 and counter > 1:
                save()
            if (counter - 1) % sample_every == 0 and counter > 1:
                generate_samples()

            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summary_loss),
                    feed_dict={context: sample_batch()})

            summary_log.add_summary(v_summary, counter)

            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')
        save()


def load_gpt2(sess,
              run_name="run1",
              checkpoint_dir="checkpoint",
              model_name=None,
              model_dir='models',
              multi_gpu=False):
    """Loads the model checkpoint or existing model into a TensorFlow session
    for repeated predictions.
    """

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])

    gpus = []
    if multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus)

    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver = tf.compat.v1.train.Saver(allow_empty=True)
    sess.run(tf.compat.v1.global_variables_initializer())

    if model_name:
        print('Loading pretrained model', ckpt)
    else:
        print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)


def generate(sess,
             run_name='run1',
             checkpoint_dir='checkpoint',
             model_name=None,
             model_dir='models',
             sample_dir='samples',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             top_p=0.0,
             include_prefix=True):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if prefix:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        context_tokens = enc.encode(prefix)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams,
        length=min(length, 1023 - (len(context_tokens) if prefix else 0)),
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )[:, 1:]

    if destination_path:
        f = open(destination_path, 'w')
    generated = 0
    gen_texts = []
    while generated < nsamples:
        if not prefix:
            out = sess.run(output)
        else:
            out = sess.run(output, feed_dict={
                    context: batch_size * [context_tokens]
                })
        for i in range(batch_size):
            generated += 1
            gen_text = enc.decode(out[i])
            if prefix:
                gen_text = enc.decode(context_tokens[:1]) + gen_text
            if truncate:
                truncate_esc = re.escape(truncate)
                if prefix and not include_prefix:
                    prefix_esc = re.escape(prefix)
                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                         truncate_esc)
                else:
                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                trunc_text = re.search(pattern, gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            gen_text = gen_text.lstrip('\n')
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim), end='')
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts


def generate_to_file(sess,
                     run_name='run1',
                     checkpoint_dir='checkpoint',
                     model_name=None,
                     model_dir='models',
                     truncate=None,
                     destination_path='gpt_2_gen_texts.txt',
                     sample_delim='=' * 20 + '\n',
                     prefix=None,
                     seed=None,
                     nsamples=1,
                     batch_size=1,
                     length=1023,
                     temperature=0.7,
                     top_k=0,
                     top_p=0.0,
                     include_prefix=True):
    """Generates the texts to a file.

    sample_delim separates texts: set to '' if each text is a small document.

    Adapted from https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn.py
    """

    generate(sess=sess,
             run_name=run_name,
             checkpoint_dir=checkpoint_dir,
             model_name=model_name,
             model_dir=model_dir,
             return_as_list=False,
             truncate=truncate,
             destination_path=destination_path,
             sample_delim=sample_delim,
             prefix=prefix,
             seed=seed,
             nsamples=nsamples,
             batch_size=batch_size,
             length=length,
             temperature=temperature,
             top_k=top_k,
             top_p=top_p,
             include_prefix=include_prefix)


def mount_gdrive():
    """Mounts the user's Google Drive in Colaboratory."""
    assert 'google.colab' in sys.modules, "You must be in Colaboratory to mount your Google Drive"

    drive.mount('/content/drive')


def is_mounted():
    """Checks if the Google Drive is mounted."""
    assert os.path.isdir('/content/drive'), "You must mount first using mount_gdrive()"


def get_tarfile_name(checkpoint_folder):
    """Converts a folder path into a filename for a .tar archive"""
    tarfile_name = checkpoint_folder.replace(os.path.sep, '_') + '.tar'

    return tarfile_name


def copy_checkpoint_to_gdrive(run_name='run1', copy_folder=False):
    """Copies the checkpoint folder to a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join('checkpoint', run_name)

    if copy_folder:
        shutil.copytree(checkpoint_folder, "/content/drive/My Drive/" + checkpoint_folder)
    else:
        file_path = get_tarfile_name(checkpoint_folder)

        # Reference: https://stackoverflow.com/a/17081026
        with tarfile.open(file_path, 'w') as tar:
            tar.add(checkpoint_folder)

        shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_checkpoint_from_gdrive(run_name='run1', copy_folder=False):
    """Copies the checkpoint folder from a mounted Google Drive."""
    is_mounted()

    checkpoint_folder = os.path.join('checkpoint', run_name)

    if copy_folder:
        shutil.copytree("/content/drive/My Drive/" + checkpoint_folder, checkpoint_folder)
    else:
        file_path = get_tarfile_name(checkpoint_folder)

        shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)

        with tarfile.open(file_path, 'r') as tar:
            tar.extractall()


def copy_file_to_gdrive(file_path):
    """Copies a file to a mounted Google Drive."""
    is_mounted()

    shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_file_from_gdrive(file_path):
    """Copies a file from a mounted Google Drive."""
    is_mounted()

    shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)


def is_gpt2_downloaded(model_dir='models', model_name='124M'):
    """Checks if the original model + associated files are present in folder."""

    for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                     'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                     'model.ckpt.meta', 'vocab.bpe']:
        if not os.path.isfile(os.path.join(model_dir, model_name, filename)):
            return False
    return True


def encode_csv(csv_path, out_path='csv_encoded.txt', header=True,
               start_token="<|startoftext|>",
               end_token="<|endoftext|>"):
    """Encodes a single-column CSV to a format suitable for gpt-2-simple.
       Automatically adds the specified prefix and suffix tokens.
    """

    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f:
        with open(out_path, 'w', encoding='utf8', errors='ignore') as w:
            if header:
                f.readline()
            reader = csv.reader(f)
            for row in reader:
                w.write(start_token + row[0] + end_token + "\n")


def encode_dataset(file_path, model_dir='models', out_path='text_encoded.npz',
                   model_name="124M",
                   combine=50000):
    """Preencodes a text document into chunks and compresses it,
    saving time when generated.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/encode.py
    """

    model_path = os.path.join(model_dir, model_name)
    enc = encoder.get_encoder(model_path)
    print('Reading files')
    chunks = load_dataset(enc, file_path, combine)
    print('Writing', out_path)
    np.savez_compressed(out_path, *chunks)


def cmd():
    """Function called when invoking from the terminal."""

    parser = argparse.ArgumentParser(
        description="Easily retrain OpenAI's GPT-2 text-generating model on new texts. (https://github.com/minimaxir/gpt-2-simple)"
    )

    # Explicit arguments
    
    parser.add_argument(
        '--mode', help='Mode for using the CLI (either "finetune" or "generate") [Required]', nargs='?')
    parser.add_argument(
        '--run_name',  help="[finetune/generate] Run number to save/load the model",
        nargs='?', default='run1')
    parser.add_argument(
        '--checkpoint_dir', help="[finetune] Path of the checkpoint directory",
        nargs='?', default='checkpoint')
    parser.add_argument(
        '--model_name',  help="[finetune] Name of the GPT-2 model to finetune",
        nargs='?', default='124M')
    parser.add_argument(
        '--model_dir', help="[finetune] Path of directory of the GPT-2 model to finetune",
        nargs='?', default='models')
    parser.add_argument(
        '--dataset',  help="[finetune] Path to the source text.",
        nargs='?', default=None)
    parser.add_argument(
        '--steps',  help="[finetune] Number of steps to train (-1 for infinite)",
        nargs='?', default=-1)
    parser.add_argument(
        '--restore_from',  help="[finetune] Whether to load model 'fresh' or from 'latest' checkpoint.",
        nargs='?', default='latest')
    parser.add_argument(
        '--sample_every',  help="[finetune] After how many steps to print sample",
        nargs='?', default=1000000, type=int)
    parser.add_argument(
        '--save_every',  help="[finetune] After how many steps to save checkpoint",
        nargs='?', default=100, type=int)
    parser.add_argument(
        '--print_every',  help="[finetune] After how many steps to print progress",
        nargs='?', default=10, type=int)
    parser.add_argument(
        '--optimizer',  help="[finetune] Optimizer to use for finetuning (adam or sgd)",
        nargs='?', default='adam')
    parser.add_argument(
        '--overwrite',  help="[finetune] Overwrite existing model when continuing training",
        nargs='?', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--nfiles',  help="[generate] How many files to generate.",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--nsamples',  help="[generate] How many texts to generate.",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--folder',  help="[generate] Folder to save the generated files",
        nargs='?', default="gen", type=str)
    parser.add_argument(
        '--length',  help="[generate] Length (tokens) of the generated texts",
        nargs='?', default=1023, type=int)
    parser.add_argument(
        '--temperature',  help="[generate] Temperature of the generated texts",
        nargs='?', default=0.7, type=float)
    parser.add_argument(
        '--top_k',  help="[generate] Sample only from top k tokens",
        nargs='?', default=0, type=int)
    parser.add_argument(
        '--top_p',  help="[generate] Sample from top p prob (overrides top_k if nonzero)",
        nargs='?', default=0.0, type=float)
    parser.add_argument(
        '--batch_size',  help="[generate] Batch size for generation (increase for GPUs)",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--prefix',  help="[generate] Prefix for generated texts",
        nargs='?', default=None)
    parser.add_argument(
        '--truncate',  help="[generate] Truncation for generated texts",
        nargs='?', default=None)
    # https://stackoverflow.com/a/46951029
    parser.add_argument(
        '--include_prefix',  help="[generate] Include prefix when truncating.",
        nargs='?', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--sample_delim',  help="[generate] Delimiter between each generated sample.",
        nargs='?', default='=' * 20 + '\n', type=str)
    parser.add_argument(
        '--multi_gpu',  help="[generate/finetune] Attempt to allocate multiple GPUs for running.",
        nargs='?', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Positional arguments
    parser.add_argument('mode', nargs='?')
    parser.add_argument('dataset', nargs='?')

    args = parser.parse_args()
    assert args.mode in ['finetune', 'generate'], "Mode must be 'finetune' or 'generate'"

    if args.mode == 'finetune':
        assert args.dataset is not None, "You need to provide a dataset."

        cmd_finetune(dataset=args.dataset, run_name=args.run_name,
                     checkpoint_dir=args.checkpoint_dir,
                     model_name=args.model_name,
                     model_dir=args.model_dir,
                     steps=args.steps, restore_from=args.restore_from,
                     sample_every=args.sample_every,
                     save_every=args.save_every,
                     print_every=args.print_every,
                     optimizer=args.optimizer,
                     overwrite=args.overwrite,
                     multi_gpu=args.multi_gpu)
    if args.mode == "generate":
        cmd_generate(nfiles=args.nfiles, nsamples=args.nsamples,
                     folder=args.folder, length=args.length,
                     temperature=args.temperature, batch_size=args.batch_size,
                     prefix=args.prefix, truncate=args.truncate,
                     include_prefix=args.include_prefix,
                     sample_delim=args.sample_delim, run_name=args.run_name,
                     checkpoint_dir=args.checkpoint_dir,
                     top_k=args.top_k, top_p=args.top_p, multi_gpu=args.multi_gpu)


def cmd_finetune(dataset, run_name, checkpoint_dir, model_name, model_dir, steps,
                 restore_from, sample_every,
                 save_every, print_every, optimizer, overwrite, multi_gpu):
    """Wrapper script for finetuning the model via the CLI."""

    if not is_gpt2_downloaded(model_dir=model_dir, model_name=model_name):
        download_gpt2(model_dir=model_dir, model_name=model_name)

    sess = start_tf_sess()
    finetune(sess, dataset=dataset, run_name=run_name,
             checkpoint_dir=checkpoint_dir,
             model_name=model_name,
             model_dir=model_dir,
             steps=steps, restore_from=restore_from,
             sample_every=sample_every, save_every=save_every,
             print_every=print_every,
             optimizer=optimizer,
             overwrite=overwrite,
             multi_gpu=multi_gpu)


def cmd_generate(nfiles, nsamples, folder,
                 length, temperature, batch_size,
                 prefix, truncate, include_prefix,
                 sample_delim, run_name,
                 checkpoint_dir,
                 top_k, top_p, multi_gpu):
    """Wrapper script for generating text via the CLI.
    The files are generated into a folder, which can be downloaded
    recursively by downloading the entire folder.
    """

    sess = start_tf_sess()
    load_gpt2(sess, run_name=run_name, checkpoint_dir=checkpoint_dir, multi_gpu=multi_gpu)

    try:
        os.mkdir(folder)
    except:
        shutil.rmtree(folder)
        os.mkdir(folder)

    for _ in trange(nfiles):
        gen_file = os.path.join(folder,
                    'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow()))

        generate_to_file(sess,
                         run_name=run_name,
                         checkpoint_dir=checkpoint_dir,
                         destination_path=gen_file,
                         length=length,
                         temperature=temperature,
                         nsamples=nsamples,
                         batch_size=batch_size,
                         prefix=prefix,
                         truncate=truncate,
                         include_prefix=include_prefix,
                         sample_delim=sample_delim,
                         top_k=top_k,
                         top_p=top_p
                         )
