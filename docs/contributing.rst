Contributing to ProgLearn
************************************

(adopted from scikit-learn)

Submitting a bug report or a feature request
--------------------------------------------

We use GitHub issues to track all bugs and feature requests; feel free to open
an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`Issue Tracker <https://github.com/neurodata/ProgLearn/issues>`_. You are
also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/neurodata/ProgLearn/issues?q=>`_
   or `pull requests <https://github.com/neurodata/ProgLearn/pulls?q=>`_.

-  If you are submitting a bug report, we strongly encourage you to follow the
   guidelines in :ref:`filing_bugs`.

.. _filing_bugs:

How to make a good bug report
-----------------------------

When you submit an issue to `Github
<https://github.com/neurodata/ProgLearn/issues>`__, please do your best to
follow these guidelines! This will make it a lot easier to provide you with
good feedback:

- The ideal bug report contains a **short reproducible code snippet**, this way
  anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/mcve>`_ for more details). If your snippet is
  longer than around 50 lines, please link to a `gist
  <https://gist.github.com>`_ or a github repo.

- If not feasible to include a reproducible snippet, please be specific about
  what **classes and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python and ProgLearn versions**. This information
  can be found by running the following code snippet::

    import platform; print(platform.platform())
    import sys; print("Python", sys.version)
    import proglearn; print("ProgLearn", proglearn.__version__)

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See `Creating and highlighting code blocks
  <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
  for more details.

Contributing Code
-----------------

The preferred workflow for contributing to `ProgLearn` is to fork the staging
repository on GitHub, clone, and develop on a branch. Steps:

1. Fork the `project repository <https://github.com/neurodata/ProgLearn>`__ by clicking
   on the ‘Fork’ button near the top right of the page. This creates a copy
   of the code under your GitHub user account. For more details on how to
   fork a repository see `this
   guide <https://help.github.com/articles/fork-a-repo/>`__.

2. Clone your fork of the ``ProgLearn`` repo from your GitHub account to your
   local disk:

   .. code:: bash

      $ git clone git@github.com:YourLogin/ProgLearn.git
      $ cd ProgLearn

3. Create a ``feature`` branch to hold your development changes:

   .. code:: bash

      $ git checkout -b my-feature

   Always use a ``feature`` branch forked from ``staging``. It’s good practice to never work on
   the raw ``main`` or ``staging`` branches!

4. Develop the feature on your feature branch. Add changed files using
   ``git add`` and then ``git commit`` files:

   .. code:: bash

      $ git add modified_files
      $ git commit

   to record your changes in Git, then push the changes to your GitHub
   account with:

   .. code:: bash

      $ git push -u origin my-feature

Pull Request Checklist
----------------------

We recommended that your contribution complies with the following rules 
(which are a brief summary of `The Bits and Brains PR Checklist <https://bitsandbrains.io/2020/10/05/pr-checklist>`__)
before you submit a pull request:

-  Follow the `coding-guidelines <#coding-guidelines>`__.
-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases ``Fix <ISSUE TITLE>`` is enough.
   ``Fix #<ISSUE NUMBER>`` is not enough.
-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.
-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.
-  All functions and classes must have unit tests. These should include,
   at the very least, type checking and ensuring correct computation/outputs.
-  Ensure all tests are passing locally using ``pytest``. Install the necessary
   packages by:

   .. code:: bash

      $ pip install pytest pytest-cov

   then run

   .. code:: bash

      $ pytest

   or you can run pytest on a single test file by

   .. code:: bash

      $ pytest path/to/test.py

-  Run an autoformatter. We use ``black`` and would like for you to
   format all files using ``black``. You can run the following lines to
   format your files.

   .. code:: bash

      $ pip install black
      $ black path/to/module.py

- PR into ``staging``. In this PR, link relevant issues (either via the use of `closing keywords <https://docs.github.com/en/enterprise/2.16/user/github/managing-your-work-on-github/closing-issues-using-keywords/>`__ in the comment or by directly linking relevant issues on the lower righthand side of the PR from the web interface), summarize the PR in the title, and comment on the PR with the following format:

  .. code:: bash

      #### Reference issue
      <Example: Closes gh-WXYZ>

      #### Type of change
      <Bug, Documentation, Feature Request>

      #### What does this implement/fix?
      <Please explain your changes>

      #### Additional information
      <Any additional information you think is important>


Coding Guidelines
-----------------

Uniformly formatted code makes it easier to share code ownership. ``ProgLearn``
package closely follows the official Python guidelines detailed in
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ that detail how
code should be formatted and indented. Please read it and follow it.

Docstring Guidelines
--------------------

Properly formatted docstrings are required for documentation generation
by Sphinx. ``ProgLearn`` package closely follows the numpydoc
guidelines. Please read and follow the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__
guidelines. Refer to the
`example.py <https://numpydoc.readthedocs.io/en/latest/example.html#example>`__
provided by numpydoc.

Tutorial Guidelines
--------------------

Properly formatted Jupyter notebooks are required for Netlify deployment.
It is recommended to check that your tutorial completes the following
steps before submitting:

- Add your notebook name to `docs/tutorials.rst
  </docs/tutorials.rst>`_ if applicable.

- Organize local functions into a separate file and put it in
  `docs/tutorials/functions </docs/tutorials/functions>`_.
  This function file and the notebook should have the same name.

- Make your tutorial self-contained if possible. It should neither
  output any file nor read from any file.

- Format the first markdown line with "#" at front and make it the
  **one and only** informative title instead of using "overview"
  or other general terms. This line will show up under the
  `tutorials <https://proglearn.neurodata.io/tutorials.html>`_ menu.

- Format the subtitles with "##" at front so that
  they will show up as submenus on website.

- Minimize cell outputs and avoid unnecessary prints.

- Remove all empty cells.

- Check the Netlify deployment preview for your PR, which is
  included as one of the required checks. Be aware that what
  you see on GitHub or local machine might be different from
  Netlify deployment, especially math formulas.
