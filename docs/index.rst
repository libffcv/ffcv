.. ffcv documentation master file, created by
   sphinx-quickstart on Sun Nov  7 17:08:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FFCV's documentation!
================================

View `Homepage <https://ffcv.io>`_ or on `GitHub <https://github.com/MadryLab/ffcv>`_.

Install ``ffcv`` via ``conda``:  ``conda install ffcv``

``ffcv`` is a package that we (students in the `MadryLab <https://madry-lab.ml>`_) created to
make training machine learning models *fast* and *easy* to use.
Below we provide a detailed walkthrough of complete examples, basics of using FFCV, advanced customizations, as well as benchmarks on ImageNet.

Walkthroughs
------------

.. toctree::
   examples
   basics
   performance_guide
   benchmarks
   customizing
   :maxdepth: 2


API Reference
-------------

We provide an API reference where we discuss the role of each module and
provide extensive documentation.

.. toctree::
   api_reference


Citation
--------
If you use this library in your research, cite it as
follows:

.. code-block:: bibtex

   @misc{leclerc2022ffcv,
      author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
      title = {ffcv},
      year = {2022},
      howpublished = {\url{https://github.com/MadryLab/ffcv/}},
      note = {commit xxxxxxx}
   }

*(Have you used the package and found it useful? Let us know!)*.


Contributors
-------------
- `Guillaume Leclerc <https://twitter.com/gpoleclerc>`_
- `Andrew Ilyas <https://twitter.com/andrew_ilyas>`_
- `Logan Engstrom <https://twitter.com/logan_engstrom>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
