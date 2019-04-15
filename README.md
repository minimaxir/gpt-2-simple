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

An example for downloading the model to the local system, fineturning it on a dataset. and generating some text.

Warning: the pretrained model, and thus any finetuned model, is 500MB!

```python
import gpt_2_simple as gpt2

gpt2.download_gpt2()   # model is saved into current directory under /117M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 'shakespeare.txt', steps=100)   # steps is max number of training steps

text = gpt2.generate(sess)
```

The generated model checkpoints are by default in `checkpoint/run1`. If you want to load a model from that folder and generate text from it:

```python
import gpt_2_simple as gpt2

gpt2.download_gpt2()   # model is saved into current directory under /117M/

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

text = gpt2.generate(sess)
```

NB: *Restart the Python session first* if you want to finetune on another dataset or load another model.

## Helpful Tips

* If you want to set a `prefix` when generating text so the generated text starts with a specific phrase, make sure to start that prefix itself with `<|endoftext|>`. (e.g. if you want to start with "Hello there", pass `<|endoftext|>Hello there`)

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## License

MIT

## Disclaimer

This repo has no affiliation or relationship with OpenAI.