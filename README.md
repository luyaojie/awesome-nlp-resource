Awesome Resource for NLP [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
====

Table of Contents
----

- __[Dictionary ](#dictionary)__
- __[Lexicon ](#lexicon)__
- __[TreeBank ](#treebank)__
- __[Language Model ](#languagemodel)__
- __[Machine Translation ](#machinetranslation)__
- __[Sentiment ](#sentment)__
- __[Question Answer ](#questionanswer)__
- __[Evaluation Dataset ](#evaluationdataset)__
- __[Word Embedding ](#wordembedding)__
- __[Other ](#other)__
- __[Event](#event)__
- __[Reference ](#reference)__

Dictionary
----
- Bilingual Dictionary
  - [CC-CEDICT](https://cc-cedict.org/wiki/start) A bilingual dictionary between English and Chinese.
- Pronouncing Dictionary
  - [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) The Carnegie Mellon University Pronouncing Dictionary is an open-source machine-readable pronunciation dictionary for North American English that contains over 134,000 words and their pronunciations. 

Lexicon
----
  - [PDEV](http://pdev.org.uk) Pattern Dictionary of English Verbs. 
  - [VerbNet](http://verbs.colorado.edu/~mpalmer/projects/verbnet.html) A lexicon that groups verbs based on their semantic/syntactic linking behavior.
  - [FrameNet](http://framenet.icsi.berkeley.edu) A lexicon based on frame semantics.
  - [WordNet](http://wordnet.princeton.edu) A lexicon that describes semantic relationships (such as synonymy and hyperonymy) between individual words.
  - [PropBank](http://en.wikipedia.org/wiki/PropBank) A corpus of one million words of English text, annotated with argument role labels for verbs; and a lexicon defining those argument roles on a per-verb basis.
  - [SemLink](https://verbs.colorado.edu/semlink) A project whose aim is to link together different lexical resources via set of mappings. (VerbNet, PropBank, FrameNet, WordNet)

TreeBank
----
  - [PTB](https://catalog.ldc.upenn.edu/ldc99t42) The Penn Treebank (PTB).
  - [Universal Dependencies](http://universaldependencies.org) Universal Dependencies (UD) is a framework for cross-linguistically consistent grammatical annotation and an open community effort with over 200 contributors producing more than 100 treebanks in over 60 languages.

Language Model
----
  - [PTB](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data) Penn Treebank Corpus in LM Version.
  - [Google Billion Word dataset](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) 1 billion word language modeling benchmark.
  - [WikiText](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger. 
  - ​

Machine Translation
----
  - [Europarl](http://www.statmt.org/europarl) The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.
  - [UNCorpus](https://conferences.unite.un.org/UNCorpus) The United Nations Parallel Corpus v1.0 is composed of official records and other parliamentary documents of the United Nations that are in the public domain.
  - [CWMT](http://nlp.nju.edu.cn/cwmt-wmt/)  The Zh-EN data collected and shared by China Workshop on Machine Translation (CWMT) community. There are three types of data for Chinese-English machine translation: Monolingual Chinese text, Parallel Chinese-English text, Multiple-Reference text.
  - [WMT](http://www.statmt.org/wmt16/translation-task.html#download) Monolingual language model training data, such as Common Crawl\News Crawl in CS\DE\EN\FI\RO\RU\TR and Parallel data.

Word Embedding
--------------
  - [Google News Word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) The model contains 300-dimensional vectors for 3 million words and phrases which trained on part of Google News dataset (about 100 billion words).
  - [GloVe Pre-trained](https://nlp.stanford.edu/projects/glove/) Pre-trained word vectors using GloVe. Wikipedia + Gigaword 5, Common Crawl, Twitter.
  - [fastText Pre-trained](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)  Pre-trained word vectors for 294 languages, trained on Wikipedia using fastText.
  - [BPEmb](https://github.com/bheinzerling/bpemb) BPEmb is a collection of pre-trained **subword embeddings** in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. 
  - [Dependency-based Word Embedding](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) Pre-trained word embeddings based on **Dependency** information, from *Dependency-Based Word Embeddings, ACL 2014.*.
  - [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/) performs ensembles of some pretrained word embedding versions, from *Meta-Embeddings: Higher-quality word embeddings via ensembles of Embedding Sets, ACL 2016.*
  - [LexVec](https://github.com/alexandres/lexvec) Pre-trained Vectors based on the **LexVec word embedding model**. Common Crawl, English Wikipedia and NewsCrawl.
  - [MUSE](https://github.com/facebookresearch/MUSE) MUSE is a Python library for multilingual word embeddings, which provide multilingual embeddings for 30 languages and 110 large-scale ground-truth bilingual dictionaries .

Question Answer
----
- Text Retrieval
  - [Ask Ubuntu](https://github.com/taolei87/askubuntu) This repo contains a preprocessed collection of questions taken from AskUbuntu.com 2014 corpus dump. It also comes with 400\*20 mannual annotations, marking pairs of questions as "similar" or "non-similar"<cite>[2]</cite>.

Other
----
  - [QA-SRL](https://dada.cs.washington.edu/qasrl/) This dataset use question-answer pairs to model verbal predicate-argument structure. The questions start with wh-words (Who, What, Where, What, etc.) and contains a verb predicate in the sentence; the answers are phrases in the sentence.


Event
----
- Event Extraction
  - [TempEval-3](https://www.cs.york.ac.uk/semeval-2013/task1/index.html) The TempEval-3 shared task aims to advance research on temporal information processing.
  - [UW Event Factuality Dataset](https://bitbucket.org/kentonl/factuality-data/src) This dataset contains annotations of text from the TempEval-3 corpus with factuality assessment labels.
  - [FactBank 1.0](https://catalog.ldc.upenn.edu/ldc2009t23) FactBank 1.0, consists of 208 documents (over 77,000 tokens) from newswire and broadcast news reports in which event mentions are annotated with their degree of factuality,
  - [ACE 2005 Training Data](http://catalog.ldc.upenn.edu/LDC2006T06) The corpus consists of data of various types annotated for entities, relations and events was created by Linguistic Data Consortium with support from the ACE Program, across three languages: English, Chinese, Arabic.
  - [Chinese Emergency Corpus (CEC)](https://github.com/shijiebei2009/CEC-Corpus) Chinese Emergency Corpus (CEC) is built by Data Semantic Laboratory in Shanghai University. This corpus is divided into 5 categories – earthquake, fire, traffic accident, terrorist attack and intoxication of food.

- Event Coreference
  - [ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus) The ECB+ corpus is an extension to the EventCorefBank.

Evaluation Dataset
----
- Event-Representation/Event Schema Induction/Script Learning
  - [Event Tensor](https://github.com/StonyBrookNLP/event-tensors/tree/master/data) A evaluation dataset about Schema Generation/Sentence Similarity/Narrative Cloze, which is proposed by <cite>Weber et al., (2018)[1]</cite>. 
- SemEval
  - [SemEval-2016 Task 9](https://github.com/HIT-SCIR/SemEval-2016) SemEval-2016 Task 9 (Chinese Semantic Dependency Parsing) Datasets

Reference
----

[1] Noah Weber, Niranjan Balasubramanian, and Nathanael Chambers. Event Representations with Tensor-based Compositions. In Proc of AAAI 2018.

[2] Tao Lei, Hrishikesh Joshi, Regina Barzilay, Tommi Jaakkola, Katerina Tymoshenko, Alessandro Moschitti, Lluis Marquez. Semi-supervised Question Retrieval with Gated Convolutions. In Proc of NAACL 2016

-----
License
----

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)