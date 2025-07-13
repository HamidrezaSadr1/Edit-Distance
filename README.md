*****
## Edit-Distance
*****

[<img src="https://img.shields.io/pypi/dm/tweet-preprocessor.svg">](https://test.pypi.org/project/edit-distance-package-urmia-university/0.0.2/#description)

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

    >>> import Edit-Distance as ed
    >>> string1 = "karolin"
    >>> string2 = "kathrin"
    >>> distance = hamming_distance(string1, string2)
    >>> print(f"The Hamming distance between '{string1}' and '{string2}' is {distance}.")
    'The Hamming distance between 'karolin' and 'kathrin' is 2'
     
    
Jacard Similarity:
-----------

     >>> string1 = "hello"
     >>> string2 = "hallo"
     >>> n_value = 2
     >>> jaccard_similarity(string1, string2, n_value)
     0.33

Categories
============ 
[<img src="https://github.com/HamidrezaSadr1/Edit-Distance/blob/main/digram.png">](https://hkiokhio2@gmail.com)


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
- send message in GitHub
- send mail to [hkiokhio2@gmail.com]
