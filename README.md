*****
Edit-Distance
*****

.. image:: https://img.shields.io/pypi/dm/tweet-preprocessor.svg
    :target: https://test.pypi.org/project/edit-distance-package-urmia-university/0.0.2/#description

Edit-Distance is a  library(package) that contain many of methods for compute edit distance between english(latin) words and 
written in Python. When you need to compare two words based on thier chars, this library makes it easy to perform without write 
same functions.

Functions
========

Currently supports this functions:

-  Hamming
-  Levenshtein
-  Damerau Levenshtein
-  Jaro
-  Jaro Winkler
-  Smileys
-  Longest Common Subsequence
-  Longest Common Substring
-  Q-gram
-  Dice Coefficient
-  Jaccard
-  Bag
-  Edit Distance with Dice Coefficient
-  Smith Waterman
-  Editex
-  Align Syllables
-  Geshtal
  
edit-distance-package-urmia-university ``v0.0.2`` supports
``Python 3.8+ on Linux, macOS and Windows``.

Usage (Examples)
=====

Hamming Distance:
---------------

.. code:: python

    >>> import Edit-Distance as ed
    >>> string1 = "karolin"
    >>> string2 = "kathrin"
    >>> distance = hamming_distance(string1, string2)
    >>> print(f"The Hamming distance between '{string1}' and '{string2}' is {distance}.")
    'The Hamming distance between 'karolin' and 'kathrin' is 2'

Jacard Similarity:
-----------

.. code:: python

     >>> string1 = "hello"
     >>> string2 = "hallo"
     >>> n_value = 2
     >>> jaccard_similarity(string1, string2, n_value)
     0.33


Available Options:
~~~~~~~~~~~~~~~~~~

+------------------+---------------------+
| Option Name      | Option Short Code   |
+==================+=====================+
| URL              | p.OPT.URL           |
+------------------+---------------------+
| Mention          | p.OPT.MENTION       |
+------------------+---------------------+
| Hashtag          | p.OPT.HASHTAG       |
+------------------+---------------------+
| Reserved Words   | p.OPT.RESERVED      |
+------------------+---------------------+
| Emoji            | p.OPT.EMOJI         |
+------------------+---------------------+
| Smiley           | p.OPT.SMILEY        |
+------------------+---------------------+
| Number           | p.OPT.NUMBER        |
+------------------+---------------------+

Installation
============

Using pip:

.. code:: bash

    $ pip install -i https://test.pypi.org/simple/ edit-distance-package-urmia-university==0.0.2


Using Anaconda:

.. code:: bash
    
    $ conda install -c [saidozcan tweet-preprocessor](https://test.pypi.org/simple/ edit-distance-package-urmia-university==0.0.2)


Comments and Suggestions
============
Use the library and give us feedback:
- On GitHub
- Mail to .. image:: data:image/png;base64,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

    :target: hkiokhio2@gmail.com

