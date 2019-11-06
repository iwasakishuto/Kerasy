# Kerasy
I want to **deepen my understanding of deep learning** by imitating the sophisticated neural networks API, **Keras**.

## Keras
![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
[![Build Status](https://travis-ci.org/keras-team/keras.svg?branch=master)](https://travis-ci.org/keras-team/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

>Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

## How to generate the articles.
1. Generate the article by [`pelican`](https://docs.getpelican.com/en/stable/).
    - Tree
    ```sh
    .Kerasy
    ├── pelican # Run `make html` HERE!!
        ├── Makefile
        ├── pelicanconf.py
        ├── publishconf.py
        ├── backdrop # Pelican Theme. (Not important.)
        ├── pelican-works # What you made by Pelican is here.
        ├── pelican-src
            ├── category1
            │   ├── markdown.md
            │   ├── jupyter.ipynb
            │   └── jupyter.ipynb-meta
            ├── category2
            │   ├── markdown.md
    ```
    - NOTE: article name (`markdown.md`) and Slug(`markdown`) must be the same.
    ```sh
    Title: CNN
    Slug: hoge
    Date: 2019-12-31 23:59
    ```
2. Make a `mkdocs.yml` for [`MkDocs`](https://www.mkdocs.org/) by running the `pelican2mkdocs.py`
    - Paste from `MkDocs/MkDocs-important/yml-templates.yml`
    - Make a **navigation Hierarchical structure** by `pelican-src` directory structure.
3. Generate the articles by `mkdocs build`.

**※ A program that performs these operations collectively is [`GithubKerasy.sh`](https://github.com/iwasakishuto/iwasakishuto.github.io/blob/master/ShellScripts/GithubKerasy.sh).**
