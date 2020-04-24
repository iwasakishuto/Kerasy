

.. image:: https://badge.fury.io/py/kerasy.svg
   :target: https://pypi.org/project/kerasy/0.0.1/
   :alt: PyPI version


.. image:: https://badge.fury.io/gh/iwasakishuto%2Fkerasy.svg
   :target: https://github.com/iwasakishuto/Kerasy
   :alt: GitHub version


.. image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000
   :target: https://github.com/iwasakishuto/Kerasy/blob/gh-pages/LICENSE
   :alt: license


Kerasy
======

I want to **deepen my understanding of deep learning** by imitating the sophisticated neural networks API, **Keras**.

Keras
^^^^^

..

   .. image:: https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png
      :target: https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png
      :alt: Keras logo


   .. image:: https://travis-ci.org/keras-team/keras.svg?branch=master
      :target: https://travis-ci.org/keras-team/keras
      :alt: Build Status


   .. image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000
      :target: https://github.com/keras-team/keras/blob/master/LICENSE
      :alt: license

   Keras is a high-level neural networks API, written in Python and capable of running on top of `TensorFlow <https://github.com/tensorflow/tensorflow>`_\ , `CNTK <https://github.com/Microsoft/cntk>`_\ , or `Theano <https://github.com/Theano/Theano>`_. It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*


How to generate the articles.
-----------------------------

.. code-block:: sh

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


#. Prepare articles (\ ``.md`` or ``.ipynb``.) NOTE: article name (\ ``XXX.md``\ ) and Slug(\ ``YYY``\ ) must be the same.(XXX=YYY)
#. Generate the html article by `\ ``pelican`` <https://docs.getpelican.com/en/stable/>`_.
   .. code-block:: sh

       # @Kerasy/pelican
       $ make html # pelican-src(.md, .ipynb) → pelican-works (.html)

#. Move html files (made by pelican) to ``MkDocs-src`` as a ``.md`` style.
#. Make a ``mkdocs.yml`` file

   * Paset from ``yml-templates.yml``
   * Get information from the Hierarchical structure of ``pelican-src``.
     .. code-block::

          # @Kerasy
          $ python pelican2mkdocs

#. Generate the articles by ``mkdocs build``.
   .. code-block::

       # @Kerasy/MkDocs
       $ mkdocs build # MkDocs-src(.md) → site (.html)

#. Copy some important static files (at ``MkDocs-important``\ ) to site dir
#. Move ``MkDocs/site`` to ``doc``.

**※ A program that performs these operations collectively is `\ ``GithubKerasy.sh`` <https://github.com/iwasakishuto/iwasakishuto.github.io/blob/master/ShellScripts/GithubKerasy.sh>`_.**

Upload to PyPI
--------------

Create your account : `https://pypi.org/ <https://pypi.org/>`_

.. code-block::

   # [Library packaging]
   # Normal. (source distribution.)
   # $ python setup.py sdist
   # wheel version. (Recommended.)
   $ python setup.py bdist_wheel

   # [Upload to PyPI]
   $ twine upload dist/*
