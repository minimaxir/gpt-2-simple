import os
import json
import requests
import sys
import shutil
import re
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
import csv

# if in Google Colaboratory
try:
    from google.colab import drive
except:
    pass

from gpt_2_simple.src import model, sample, encoder
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer


def download_gpt2(model_name='117M'):
    """Downloads the GPT-2 model into the current directory
    from Google Cloud Storage.

    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    """

    subdir = os.path.join('models', model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                     'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                     'model.ckpt.meta', 'vocab.bpe']:

        r = requests.get("https://storage.googleapis.com/gpt-2/" +
                         subdir + "/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename,
                      total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def start_tf_sess():
    """
    Returns a tf.Session w/ config
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def finetune(sess,
             dataset,
             steps=-1,
             model_name='117M',
             combine=50000,
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             restore_from='latest',
             run_name='run1',
             sample_every=100,
             sample_length=1023,
             sample_num=1,
             save_every=1000,
             print_every=1,
             max_checkpoints=1,
             model_load=False):
    """Finetunes the model on the given dataset.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    if not model_load:
        for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
            shutil.copyfile(os.path.join('models', model_name, file),
                            os.path.join(checkpoint_path, file))

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=sample_length,
        context=context,
        batch_size=batch_size,
        temperature=1.0,
        top_k=40)

    train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    if accumulate_gradients > 1:
        opt = AccumulatingOptimizer(
            opt=tf.train.AdamOptimizer(learning_rate=learning_rate),
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        opt_apply = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
                loss, var_list=train_vars)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_log = tf.summary.FileWriter(checkpoint_path)

    saver = tf.train.Saver(
        var_list=train_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join('models', model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    if model_load:
        return

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter) + '\n')

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

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if counter == steps:
                save()
                return
            if counter % save_every == 0:
                save()
            if counter % sample_every == 0:
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
              run_name="run1"):
    """Loads the model checkpoint into a TensorFlow session
    for repeated predictions.
    """

    finetune(sess, '', run_name=run_name, model_load=True)


def generate(sess,
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             model_name='117M',
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             run_name='run1'):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix:
        context = tf.placeholder(tf.int32, [batch_size, None])

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    np.random.seed(seed)
    tf.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )[:, 1:]

    if destination_path:
        f = open(destination_path, 'w')
    if prefix:
        context_tokens = enc.encode(prefix)
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
                gen_text = prefix[0] + gen_text
            if truncate:
                trunc_text = re.search(r'(.*?)(?:{})'.format(truncate),
                                       gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim))
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts


def generate_to_file(sess,
                     truncate=None,
                     destination_path='gpt_2_gen_texts.txt',
                     sample_delim='=' * 20 + '\n',
                     prefix=None,
                     model_name='117M',
                     seed=None,
                     nsamples=1,
                     batch_size=1,
                     length=1023,
                     temperature=0.7,
                     top_k=0,
                     run_name='run1'):
    """Generates the texts to a file.

    sample_delim separates texts: set to '' if each text is a small document.

    Adapted from https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn.py
    """

    generate(sess,
             False,
             truncate,
             destination_path,
             sample_delim,
             prefix,
             model_name,
             seed,
             nsamples,
             batch_size,
             length,
             temperature,
             top_k,
             run_name)


def mount_gdrive():
    """Mounts the user's Google Drive in Colaboratory."""
    assert 'google.colab' in sys.modules, "You must be in Colaboratory to mount your Google Drive"

    drive.mount('/content/drive')


def is_mounted():
    """Checks if the Google Drive is mounted."""
    assert os.path.isdir('/content/drive'), "You must mount first using mount_gdrive()"


def copy_checkpoint_to_gdrive(checkpoint_folder=os.path.join('checkpoint', 'run1')):
    """Copies the checkpoint folder to a mounted Google Drive."""
    is_mounted()

    shutil.copytree(checkpoint_folder, "/content/drive/My Drive/" + checkpoint_folder)


def copy_checkpoint_from_gdrive(checkpoint_folder=os.path.join('checkpoint', 'run1')):
    """Copies the checkpoint folder from a mounted Google Drive."""
    is_mounted()

    shutil.copytree("/content/drive/My Drive/" + checkpoint_folder, checkpoint_folder)


def copy_file_to_gdrive(file_path):
    """Copies a file to a mounted Google Drive."""
    is_mounted()

    shutil.copyfile(file_path, "/content/drive/My Drive/" + file_path)


def copy_file_from_gdrive(file_path):
    """Copies a file from a mounted Google Drive."""
    is_mounted()

    shutil.copyfile("/content/drive/My Drive/" + file_path, file_path)


def is_gpt2_downloaded(model_path=os.path.join("models", "117M")):
    """Checks if the original model + associated files are present in folder."""

    for filename in ['checkpoint', 'encoder.json', 'hparams.json',
                     'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
                     'model.ckpt.meta', 'vocab.bpe']:
        if not os.path.isfile(os.path.join(model_path, filename)):
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
