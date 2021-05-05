MOKAS: a MOKe Analysis Software
===============================


MOKAS (MOKe Analysis Software) is an open-source project to analyse experimental MOKE data. It is totally written in Python, and it is available on github at github.com/gdurin/mokas. The current version is 0.7.0, which means the software is still in beta, waiting for suggestions and tests from the closest colleagues of the MagnEFi and TOPS European project. From version 0.8, scheduled in June, the software will be announced officially to a larger community.
The code makes extensive use of CUDA for parallel computing, highly improving the speed of the calculation. Several output plots are available, to better study the magnetization dynamics in magnetic bubbles and wires at the nanoscale.


The idea is to learn the essential elements of Python by trying on 'real' problems. Something a PhD student or a researcher daily finds in a lab: reading files, make calcultations, plot results, do data fitting, image analysis, etc.

To open the notebooks, download the files into a directory on your computer and from that directory run:

    $ ipython notebook

This will open a new page in your browser with a list of the available notebooks.

Should this error `[TerminalIPythonApp] WARNING | File not found: u'notebook'` pop up, please install Jupyter by following the [instructions](http://jupyter.readthedocs.io/en/latest/install.html) and execute the following command to run the notebook:

    $ jupyter notebook

The Bk.zip file is a collection of files that have to be unzipped in a subfolder (such as Bk). They are used all along the notebooks.

The notebooks have been made using Python 3.X (with an exception, as written in the file for the 2017 edition)

References
==========
* [Lectures on scientific computing with Python of Robert Johansson](https://github.com/jrjohansson/scientific-python-lectures)
* [The notebook viewer](http://nbviewer.jupyter.org/)
* [LMFIT: Non-Linear Least-Squares Minimization and Curve-Fitting for Python](https://lmfit.github.io/lmfit-py/model.html)

License
=======
This work is licensed under a [Creative Commons Attribution 3.0 Unported License.](http://creativecommons.org/licenses/by/3.0/)
