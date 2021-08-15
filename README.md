# Automatic Error Correction Using the Wikipedia Page Revision History

Error correction is one of the most crucial and time-consuming steps of data preprocessing. State-of-the-art error correction systems leverage various signals, such as predefined data constraints or user-provided correction examples, to fix erroneous values in a semi-supervised manner. While these approaches reduce human involvement to a few labeled tuples, they still need supervision to fix data errors. In this paper, we propose a novel error correction approach to automatically fix data errors of dirty datasets. Our approach pretrains a set of error corrector models on correction examples extracted from the Wikipedia page revision history. It then fine-tunes these models on the dirty dataset at hand without any required user labels. Finally, our approach aggregates the fine-tuned error corrector models to find the actual correction of each data error. As our experiments show, our approach automatically fixes a large portion of data errors of various dirty datasets with high precision. 

## Wiki Dump Files
[Wiki Dump File Link:July 2020](https://dumps.wikimedia.org/enwiki/20200701/)

## Wiki Revision Table & Infobox Parser
[mwparserfromhell](https://github.com/earwig/mwparserfromhell)

[wikitextparser](https://github.com/5j9/wikitextparser)

## Extract Old_New Values
[Difflib](https://docs.python.org/3/library/difflib.html)

## Model: Pretrained, Finetune
[Edit_Distance](http://norvig.com/spell-correct.html)

[Gensim](https://radimrehurek.com/gensim/)

[fastText](https://fasttext.cc/)



## Code Adapted 
[Raha](https://github.com/BigDaMa/raha)

[Typo Error](https://github.com/makcedward/nlp/blob/master/sample/util/nlp-util-symspell.ipynb)



