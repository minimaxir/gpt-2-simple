import os
import json
import sys
import requests
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time

from src import model, sample, encoder
from src.load_dataset import load_dataset, Sampler
from src.accumulate import AccumulatingOptimizer


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
             save_every=1000):
    """Finetunes the model on the given dataset.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    CHECKPOINT_DIR = 'checkpoint'
    SAMPLE_DIR = 'samples'

    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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

    summary_log = tf.summary.FileWriter(
        os.path.join(CHECKPOINT_DIR, run_name))

    saver = tf.train.Saver(
        var_list=train_vars,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=2)
    sess.run(tf.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(
            os.path.join(CHECKPOINT_DIR, run_name))
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

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(CHECKPOINT_DIR, run_name, 'counter')
    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1

    def save():
        maketree(os.path.join(CHECKPOINT_DIR, run_name))
        print(
            'Saving',
            os.path.join(CHECKPOINT_DIR, run_name,
                         'model-{}').format(counter))
        saver.save(
            sess,
            os.path.join(CHECKPOINT_DIR, run_name, 'model'),
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
                sys.exit()
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
              model_name="117M",
              checkpoint_path=os.path.join('models', model_name)):
    """Loads the model checkpoint into a TensorFlow session
    for repeated predictions.
    """

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, ckpt)


def generate(sess,
             prefix='<|endoftext|>',
             model_name='117M',
             seed=None,
             nsamples=1,
             batch_size=1,
             length=None,
             temperature=1,
             top_k=0):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    np.random.seed(seed)
    tf.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=enc.encoder[prefix],
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )[:, 1:]

    generated = 0
    gen_texts = []
    while generated < nsamples:
        out = sess.run(output)
        for i in range(batch_size):
            generated += batch_size
            gen_texts.append(enc.decode(out[i]))

    return gen_texts
