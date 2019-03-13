Awesome Resource for NLP [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
====

Table of Contents
----

- [Awesome Resource for NLP ![Awesome](https://github.com/sindresorhus/awesome)](#awesome-resource-for-nlp-awesomehttpsgithubcomsindresorhusawesome)
  - [Table of Contents](#table-of-contents)
  - [Dictionary](#dictionary)
  - [Lexicon](#lexicon)
  - [Parsing](#parsing)
  - [Language Model](#language-model)
  - [Machine Translation](#machine-translation)
  - [Text Generation](#text-generation)
  - [Sentiment](#sentiment)
  - [Word Representation](#word-representation)
  - [Question Answer](#question-answer)
  - [Information Extraction](#information-extraction)
  - [Natural Language Inference](#natural-language-inference)
  - [Other](#other)
  - [License](#license)

<span id='dictionary'>Dictionary</span>
----
- Bilingual Dictionary
  - [CC-CEDICT](https://cc-cedict.org/wiki/start) A bilingual dictionary between English and Chinese.
- Pronouncing Dictionary
  - [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) The Carnegie Mellon University Pronouncing Dictionary is an open-source machine-readable pronunciation dictionary for North American English that contains over 134,000 words and their pronunciations. 

<span id='lexicon'>Lexicon</span>
----
  - [PDEV](http://pdev.org.uk) Pattern Dictionary of English Verbs. 
  - [VerbNet](http://verbs.colorado.edu/~mpalmer/projects/verbnet.html) A lexicon that groups verbs based on their semantic/syntactic linking behavior.
  - [FrameNet](http://framenet.icsi.berkeley.edu) A lexicon based on frame semantics.
  - [WordNet](http://wordnet.princeton.edu) A lexicon that describes semantic relationships (such as synonymy and hyperonymy) between individual words.
  - [PropBank](http://en.wikipedia.org/wiki/PropBank) A corpus of one million words of English text, annotated with argument role labels for verbs; and a lexicon defining those argument roles on a per-verb basis.
  - [NomBank](https://nlp.cs.nyu.edu/meyers/NomBank.html)  A dataset marks the sets of arguments that cooccur with nouns in the PropBank Corpus (the Wall Street Journal Corpus of the Penn Treebank), just as PropBank records such information for verbs.
  - [SemLink](https://verbs.colorado.edu/semlink) A project whose aim is to link together different lexical resources via set of mappings. (VerbNet, PropBank, FrameNet, WordNet)
  - [Framester](https://lipn.univ-paris13.fr/framester/) Framester is a hub between FrameNet, WordNet, VerbNet, BabelNet, DBpedia, Yago, DOLCE-Zero, as well as other resources. Framester does not simply creates a strongly connected knowledge graph, but also applies a rigorous formal treatment for Fillmore's frame semantics, enabling full-fledged OWL querying and reasoning on the created joint frame-based knowledge graph.

<span id='parsing'>Parsing</span>
----
  - [PTB](https://catalog.ldc.upenn.edu/ldc99t42) The Penn Treebank (PTB).
  - [Universal Dependencies](http://universaldependencies.org) Universal Dependencies (UD) is a framework for cross-linguistically consistent grammatical annotation and an open community effort with over 200 contributors producing more than 100 treebanks in over 60 languages.
  - [SemEval-2016 Task 9](https://github.com/HIT-SCIR/SemEval-2016) SemEval-2016 Task 9 (Chinese Semantic Dependency Parsing) Datasets

<span id='lm'>Language Model</span>
----
  - [PTB](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data) Penn Treebank Corpus in LM Version.
  - [Google Billion Word dataset](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) 1 billion word language modeling benchmark.
  - [WikiText](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger. 

<span id='mt'>Machine Translation</span>
----
  - [Europarl](http://www.statmt.org/europarl) The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.
  - [UNCorpus](https://conferences.unite.un.org/UNCorpus) The United Nations Parallel Corpus v1.0 is composed of official records and other parliamentary documents of the United Nations that are in the public domain.
  - [CWMT](http://nlp.nju.edu.cn/cwmt-wmt/)  The Zh-EN data collected and shared by China Workshop on Machine Translation (CWMT) community. There are three types of data for Chinese-English machine translation: Monolingual Chinese text, Parallel Chinese-English text, Multiple-Reference text.
  - [WMT](http://www.statmt.org/wmt16/translation-task.html#download) Monolingual language model training data, such as Common Crawl\News Crawl in CS\DE\EN\FI\RO\RU\TR and Parallel data.
  - [OPUS](http://opus.nlpl.eu) OPUS is a growing collection of translated texts from the web. In the OPUS project we try to convert and align free online data, to add linguistic annotation, and to provide the community with a publicly available parallel corpus. 

<span id='textgeneration'>Text Generation</span>
----
  - [ACL Title and Abstract Dataset](https://github.com/EagleW/ACL_titles_abstracts_dataset) This dataset gathers 10,874 title and abstract pairs from the ACL Anthology Network (until 2016).
  - [WikiBio](https://github.com/DavidGrangier/wikipedia-biography-dataset) This dataset gathers 728,321 biographies from wikipedia. It aims at evaluating text generation algorithms. For each article, it provide the first paragraph and the infobox (both tokenized).
  - [Tencent Automatic Article Commenting](http://ai.tencent.com/upload/PapersUploads/article_commenting.tgz) A large-scale Chinese dataset with millions of real comments and a human-annotated subset characterizing the comments’ varying quality. This dataset consists of around 200K news articles and 4.5M human comments along with rich meta data for article categories and user votes of comments.

<span id='sentiment'>Sentiment</span>
---------
  - [MPQA 3.0](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/) This corpus contains news articles and other text documents manually annotated for opinions and other private states (i.e., beliefs, emotions, sentiments, speculations, etc.). The main changes in this version of the MPQA corpus are the additions of new eTarget (entity/event) annotations.
  - [SenticNet](http://sentic.net) SenticNet provides a set of semantics, sentics, and polarity associated with 100,000 natural language concepts. SenticNet consists of a set of tools and techniques for sentiment analysis combining commonsense reasoning, psychology, linguistics, and machine learning. 
  - [SentiWordNet](http://sentiwordnet.isti.cnr.it) SentiWordNet is a lexical resource for opinion mining. SentiWordNet assigns to each synset of WordNet three sentiment scores: positivity, negativity, objectivity.
  - [NRC Word-Emotion Association Lexicon ](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) The NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). 
  - [Stanford Sentiment TreeBank](https://nlp.stanford.edu/sentiment/index.html) SST is the dataset of the paper: Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
  - [SemEval-2013 Twitter](https://www.cs.york.ac.uk/semeval-2013/task2/index.html) SemEval 2013 Twitter dataset, which contains phrase-level sentiment annotation. 

<span id='wordrepresentation'>Word Representation</span>
--------------
- Word Embedding
  - [Google News Word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) The model contains 300-dimensional vectors for 3 million words and phrases which trained on part of Google News dataset (about 100 billion words).
  - [GloVe Pre-trained](https://nlp.stanford.edu/projects/glove/) Pre-trained word vectors using GloVe. Wikipedia + Gigaword 5, Common Crawl, Twitter.
  - [fastText Pre-trained](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)  Pre-trained word vectors for 294 languages, trained on Wikipedia using fastText.
  - [BPEmb](https://github.com/bheinzerling/bpemb) BPEmb is a collection of pre-trained **subword embeddings** in 275 languages, based on Byte-Pair Encoding (BPE) and trained on Wikipedia. 
  - [Dependency-based Word Embedding](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) Pre-trained word embeddings based on **Dependency** information, from *Dependency-Based Word Embeddings, ACL 2014.*.
  - [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/) performs ensembles of some pretrained word embedding versions, from *Meta-Embeddings: Higher-quality word embeddings via ensembles of Embedding Sets, ACL 2016.*
  - [LexVec](https://github.com/alexandres/lexvec) Pre-trained Vectors based on the **LexVec word embedding model**. Common Crawl, English Wikipedia and NewsCrawl.
  - [MUSE](https://github.com/facebookresearch/MUSE) MUSE is a Python library for multilingual word embeddings, which provide multilingual embeddings for 30 languages and 110 large-scale ground-truth bilingual dictionaries .
  - [CWV](https://github.com/Embedding/Chinese-Word-Vectors) This project provides 100+ Chinese Word Vectors (embeddings) trained with different representations (dense and sparse), context features (word, ngram, character, and more), and corpora.
  - [charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/) This repository provieds the re-implemented code for pre-training character n-gram embeddings presented in Joint Many-Task (JMT) paper, *A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks, EMNLP2017*.
- Word Representation with Context
  - [ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) Pre-trained contextual representations from large scale bidirectional language models provide large improvements for nearly all supervised NLP tasks.
  - [BERT](https://github.com/google-research/bert) **BERT**, or **B**idirectional **E**ncoder **R**epresentations from **T**ransformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. (2018.10)

<span id="qa">Question Answer</span>
----
- Machine Reading Comprehension
  - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage. 
  - [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. This dataset is from the Wikipedia domain and Web domain.
  - [NewsQA](https://datasets.maluuba.com/NewsQA) NewsQA is a crowd-sourced machine reading comprehension dataset of 120K Q&A pairs. 
  - [HarvestingQA](https://github.com/xinyadu/harvestingQA/tree/master/dataset) This folder contains the one million paragraph-level QA-pairs dataset (split into Train, Dev and Test set) described in: *Harvesting Paragraph-Level Question-Answer Pairs from Wikipedia* (ACL 2018).
- <span id="SimilarQuestionIden">Duplicate/Similar Question Identification</span>
  - [Quora Question Pairs](http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv) Quora Question Pairs dataset consists of over 400,000 lines of potential question duplicate pairs. [[Kaggle Version Format]](https://www.kaggle.com/c/quora-question-pairs/data)
  - [Ask Ubuntu](https://github.com/taolei87/askubuntu) This repo contains a preprocessed collection of questions taken from AskUbuntu.com 2014 corpus dump. It also comes with 400\*20 mannual annotations, marking pairs of questions as "similar" or "non-similar", from *Semi-supervised Question Retrieval with Gated Convolutions, NAACL2016*.


<span id="ie">Information Extraction</span>
----
- Entity
  - [Shimaoka Fine-grained](http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip) This dataset contains two standard and publicly available datasets for Fine-grained Entity Classification, provided in a preprocessed tokenized format, details in *Neural architectures for ﬁne-grained entity type classiﬁcation, EACL 2017*.
  - [Ultra-Fine Entity Typing](https://homes.cs.washington.edu/~eunsol/_site/open_entity.html) A new entity typing task: given a sentence with an entity mention, the goal is to predict a set of free-form phrases (e.g. skyscraper, songwriter, or criminal) that describe appropriate types for the target entity.
  - 

- Relation Extraction
  - [SemEval 2018 Task7](https://lipn.univ-paris13.fr/~gabor/semeval2018task7/) The training data and evaluation script for SemEval 2018 Task 7: Semantic Relation Extraction and Classification in Scientific Papers. 
  - [Datasets of Annotated Semantic Relationships](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets) **RECOMMEND** This repository contains annotated datasets which can be used to train supervised models for the task of semantic relationship extraction.

- Event Extraction
  - [TempEval-3](https://www.cs.york.ac.uk/semeval-2013/task1/index.html) The TempEval-3 shared task aims to advance research on temporal information processing.
  - [UW Event Factuality Dataset](https://bitbucket.org/kentonl/factuality-data/src) This dataset contains annotations of text from the TempEval-3 corpus with factuality assessment labels.
  - [FactBank 1.0](https://catalog.ldc.upenn.edu/ldc2009t23) FactBank 1.0, consists of 208 documents (over 77,000 tokens) from newswire and broadcast news reports in which event mentions are annotated with their degree of factuality,
  - [ACE 2005 Training Data](http://catalog.ldc.upenn.edu/LDC2006T06) The corpus consists of data of various types annotated for entities, relations and events was created by Linguistic Data Consortium with support from the ACE Program, across three languages: English, Chinese, Arabic.
  - [Chinese Emergency Corpus (CEC)](https://github.com/shijiebei2009/CEC-Corpus) Chinese Emergency Corpus (CEC) is built by Data Semantic Laboratory in Shanghai University. This corpus is divided into 5 categories – earthquake, fire, traffic accident, terrorist attack and intoxication of food.
  - [TAC-KBP](https://tac.nist.gov) Event Evaluation is a sub-track in TAC Knowledge Base Population (KBP), which started from 2015. The goal of TAC Knowledge Base Population (KBP) is to develop and evaluate technologies for populating knowledge bases (KBs) from unstructured text.

- Event-Representation/Event Schema Induction/Script Learning
  - [Narrative Cloze Evaluation Data](https://www.usna.edu/Users/cs/nchamber/data/chains)  Evaluate understanding of a script by predicting the next event given several context events. Details in *Unsupervised Learning of Narrative Schemas and their Participants, ACL 2009*.

  - [Event Tensor](https://github.com/StonyBrookNLP/event-tensors/tree/master/data) A evaluation dataset about Schema Generation/Sentence Similarity/Narrative Cloze, which is proposed by *Event Representations with Tensor-based Compositions, AAAI 2018.*. 

- Event/Time Extraction/Time Line Generation
  - [TimeBank](https://catalog.ldc.upenn.edu/LDC2006T08) TimeBank 1.2 contains 183 news articles that have been annotated with temporal information, adding events, times and temporal links(TLINKs) between events and times.
  - [TimeBank-EventTime Corpus](https://www.ukp.tu-darmstadt.de/data/timeline-generation/temporal-anchoring-of-events/) This dataset is a subset of the TimeBank Corpus with a new annotation scheme to anchor events in time. [Detailed description](https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2016/2016_Reimers_Temporal_Anchoring_of_Events.pdf).
  - [SemEval-2015 Task 4](http://alt.qcri.org/semeval2015/task4/) TimeLine: Cross-Document Event Ordering. Given a set of documents and a target entity, the task is to build an event TimeLine related  to that entity, i.e. to detect, anchor in time and order the events involving the target entity.

- Event Coreference
  - [ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus) The ECB+ corpus is an extension to the EventCorefBank.

- Open Information Extraction
  - [oie-benchmark](https://github.com/gabrielStanovsky/oie-benchmark#converting-qa-srl-to-open-ie) This repository contains code for converting QA-SRL annotations to Open-IE extractions and comparing Open-IE parsers against a converted benchmark corpus.q
  - [NeuralOpenIE](https://onedrive.live.com/?authkey=%21AHj1kHDE5TSS0e8&cid=C826C2D6F4C7D993&id=C826C2D6F4C7D993%213193&parId=C826C2D6F4C7D993%213189&action=locate) A training dataset from *Neural Open Information Extraction*, ACL 2018. here are a total of 36,247,584 hsentence, tuplei pairs extracted from Wikipedia dump using OPENIE4.

- Entity Linking
  - [WikilinksNED](https://github.com/yotam-happy/NEDforNoisyText) A large-scale Named Entity Disambiguation dataset of text fragments from the web, which is significantly noisier and more challenging than existing news-based datasets.
<span id="nli">Natural Language Inference</span>
----
  - [SNLI](https://nlp.stanford.edu/projects/snli/) The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).
  - [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus is modeled on the SNLI corpus, but differs in that covers **a range of genres** of spoken and written text, and supports a distinctive cross-genre generalization evaluation.
  - [Scitail](http://data.allenai.org/scitail/) The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. The domain makes this dataset different in nature from previous datasets, and it consists of more factual sentences rather than scene descriptions.
  - [Duplicate/Similar Question Identification](#SimilarQuestionIden)

Other
----
  - [QA-SRL](https://dada.cs.washington.edu/qasrl/) This dataset use question-answer pairs to model verbal predicate-argument structure. The questions start with wh-words (Who, What, Where, What, etc.) and contains a verb predicate in the sentence; the answers are phrases in the sentence.
  - [QA-SRL 2.0](https://github.com/uwnlp/qasrl-bank) This repository is the reference point for QA-SRL Bank 2.0, the dataset described in the paper Large-Scale QA-SRL Parsing, ACL 2018. 
  - [NEWSROOM](https://summari.es) CORNELL NEWSROOM is a large dataset for training and evaluating summarization systems. It contains 1.3 million articles and summaries written by authors and editors in the newsrooms of 38 major publications.
  - [CoNLL 2010 Uncertainty Detection](http://rgai.inf.u-szeged.hu/conll2010st/tasks.html) The aim of this task is to identify sentences in texts which contain unreliable or uncertain information. Training Data contains biological abstracts and full articles from the **BioScope** (biomedical domain) corpus and paragraphs from **Wikipedia** possibly containing weasel information.
  - [Automatic Academic Paper Rating](https://github.com/lancopku/AAPR) A dataset for automatic academic paper rating (AAPR), which automatically determine whether to accept academic papers. The dataset consists of 19,218 academic papers by collecting data on academic pa- pers in the field of artificial intelligence from the arxiv.
  - [COLING 2018 automatic identification of verbal MWE](https://gitlab.com/parseme/sharedtask-data/tree/master/1.1) Corpora were annotated by human annotators with occurrences of verbal multiword expressions (VMWEs) according to common annotation guidelines. For example, "He **picked** one **up**."

-----
License
----

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)