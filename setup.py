from setuptools import setup, find_packages

long_description = '''
A simple Python package that wraps existing model fine-tuning and generation scripts for OpenAI GPT-2 text generation model (specifically the "small", 117M hyperparameter version). Additionally, this package allows easier generation of text, generating to a file for easy curation, allowing for prefixes to force the text to start with a given phrase.

## Usage

An example for downloading the model to the local system, fineturning it on a dataset. and generating some text.

Warning: the pretrained model, and thus any finetuned model, is 500 MB!

```python
import gpt_2_simple as gpt2

gpt2.download_gpt2()   # model is saved into current directory under /models/117M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 'shakespeare.txt', steps=1000)   # steps is max number of training steps

gpt2.generate(sess)
```

The generated model checkpoints are by default in `/checkpoint/run1`. If you want to load a model from that folder and generate text from it:

```python
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess)
```

As with textgenrnn, you can generate and save text for later use (e.g. an API or a bot) by using the `return_as_list` parameter.

```python
single_text = gpt2.generate(sess, return_as_list=True)[0]
print(single_text)
```

You can pass a `run_name` parameter to `finetune` and `load_gpt2` if you want to store/load multiple models in a `checkpoint` folder.

NB: *Restart the Python session first* if you want to finetune on another dataset or load another model.
'''


setup(
    name='gpt_2_simple',
    packages=['gpt_2_simple'],  # this must be the same as the name above
    version='0.5',
    description="Python package to easily retrain OpenAI's GPT-2 " \
    "text-generating model on new texts.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Max Woolf',
    author_email='max@minimaxir.com',
    url='https://github.com/minimaxir/gpt-2-simple',
    keywords=['deep learning', 'tensorflow', 'text generation'],
    classifiers=[],
    license='MIT',
    entry_points={
        'console_scripts': ['gpt_2_simple=gpt_2_simple.gpt_2:cmd'],
    },
    python_requires='>=3.5',
    include_package_data=True,
    install_requires=['regex', 'requests', 'tqdm', 'numpy', 'toposort']
)
