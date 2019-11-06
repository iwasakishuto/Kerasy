# Kerasy
I want to **deepen my understanding of deep learning** by imitating the sophisticated neural networks API, **Keras**.

## Keras
![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
[![Build Status](https://travis-ci.org/keras-team/keras.svg?branch=master)](https://travis-ci.org/keras-team/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

>Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

## How to generate the articles.
```sh
.Kerasy
├── MkDocs
│   ├── MkDocs-important
|   |   |   ├── img
|   |   |   ├── theme
│   │   │   └── index.md
│   │   └── yml-templates.yml
│   ├── site
│   ├── MkDocs-src
│   └── mkdocs.yml
├── README.md
├── doc
├── kerasy
├── pelican
│   ├── Makefile
│   ├── backdrop
│   ├── pelican-src
│   ├── pelican-works
│   ├── pelicanconf.py
│   └── publishconf.py
└── pelican2mkdocs.py
```

0. Prepare articles (`.md` or `.ipynb`.) NOTE: article name (`XXX.md`) and Slug(`YYY`) must be the same.(XXX=YYY)
1. Generate the html article by [`pelican`](https://docs.getpelican.com/en/stable/).
```sh
# @Kerasy/pelican
$ make html # pelican-src(.md, .ipynb) → pelican-works (.html)
```
2. Move html files (made by pelican) to `MkDocs-src` as a `.md` style.
3. Make a `mkdocs.yml` file
```
# @Kerasy
$ python pelican2mkdocs
```
    - Paset from `yml-templates.yml`
    - Get information from the Hierarchical structure of `pelican-src`.
4. Generate the articles by `mkdocs build`.
```
# @Kerasy/MkDocs
$ mkdocs build # MkDocs-src(.md) → site (.html)
```
5. Copy some important static files (at `MkDocs-important`) to site dir
6. Move `MkDocs/site` to `doc`.

**※ A program that performs these operations collectively is [`GithubKerasy.sh`](https://github.com/iwasakishuto/iwasakishuto.github.io/blob/master/ShellScripts/GithubKerasy.sh).**
