# gpt-2-simple

A simple Python package that wraps existing model fine-tuning and generation scripts for [OpenAI](https://openai.com)'s [GPT-2 text generation model](https://openai.com/blog/better-language-models/) (specifically the "small", 117M hyperparameter version). Additionally, this package allows easier generation of text, allowing for prefixes, generating text of any length.

This package incorporates and makes minimal low-level changes to:

* Model management from OpenAI's [official GPT-2 repo](https://github.com/openai/gpt-2) (MIT License)
* Model finetuning from Neil Shepperd's [fork](https://github.com/nshepperd/gpt-2) of GPT-2 (MIT License)
* Text generation output management from [textgenrnn](https://github.com/minimaxir/textgenrnn) (MIT License / also created by me)

For finetuning, it is strongly recommended to use a GPU. If you are training in the cloud, using a Colaboratory notebook or a Google Compute Engine VM w/ the TensorFlow Deep Learning image is strongly recommended. (as the GPT-2 model is hosted on GCP)

## Usage

gpt-2-simple can be installed via pip:

```shell
pip3 install gpt-2-simple
```

An example for downloading the model to the local computer and finetining it on a dataset.

Warning: the pretrained model is a ~500MB download!

```python
import gpt_2_simple as gpt2
import tensorflow as tf

gpt2.download_gpt2()   # model is saved into current directory under /117M/

sess = tf.Session(graph=tf.Graph())
gpt2.finetune(sess, 'shakespeare.txt', steps=5000)   # steps is max number of training steps

gpt2.load_gpt2(sess)
text = gpt2.generate()
```

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## License

MIT

## Disclaimer

This repo has no affiliation or relationship with OpenAI.