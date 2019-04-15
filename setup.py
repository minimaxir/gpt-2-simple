from setuptools import setup, find_packages

long_description = '''
A simple Python package that wraps existing model fine-tuning and generation scripts for OpenAI's GPT-2 text generation model (specifically the "small", 117M hyperparameter version). Additionally, this package allows easier generation of text, allowing for prefixes, generating text of any length.
'''


setup(
    name='gpt_2_simple',
    packages=['gpt_2_simple'],  # this must be the same as the name above
    version='0.1',
    description='Provide an input CSV and a target field to predict, ' \
    'generate a model + code to run it.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Max Woolf',
    author_email='max@minimaxir.com',
    url='https://github.com/minimaxir/gpt-2-simple',
    keywords=['deep learning', 'tensorflow', 'text generation'],
    classifiers=[],
    license='MIT',
    python_requires='>=3.5',
    include_package_data=True,
    install_requires=['regex', 'requests', 'tqdm', 'numpy']
)
