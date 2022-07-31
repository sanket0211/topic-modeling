# Dataset Study

## Data Statistics

```bash
                                           documents
0  Tires where delivered to the garage of my choi...
1  Easy Tyre Selection Process, Competitive Prici...
2         Very easy to use and good value for money.
No. of Documents 10132
Average document length 25.463975523095144
Total no. of unique tokens 9832
```

## Plot Description
- Plot is drawn between frequency of occurrence of different document lengths. 
- We see that most of the documents have lengths between 10 to 150

![length of Document v/s Frequency of occurrence](length_of_documents-vs-frequency-plot.png?raw=true "length of Document v/s Frequency of occurrence")


## Top 100 unigrams

```bash
['tyre', 'service', 'price', 'good', 'easy', 'garage', 'great', 'fitting', 'time', 'use', 'excellent', 'fitted', 'local', 'day', 'fitter', 'quick', 'choice', 'used', 'car', 'efficient', 'order', 'value', 'get', 'website', 'recommend', 'customer', 'appointment', 'competitive', 'experience', 'friendly', 'process', 'first', 'best', 'always', 'problem', 'definitely', 'done', 'fit', 'simple', 'got', 'new', 'booking', 'online', 'convenient', 'delivered', 'really', 'well', 'helpful', 'went', 'booked', 'date', 'centre', 'work', 'job', 'ordered', 'using', 'issue', 'money', 'also', 'wheel', 'happy', 'ordering', 'way', 'staff', 'at', 'email', 'book', 'back', 'delivery', 'arrived', 'hassle', 'choose', 'recommended', 'go', 'fast', 'told', 'quality', 'company', 'professional', 'find', 'highly', 'free', 'hour', 'site', 'cheaper', 'everything', 'buy', 'purchase', 'slot', 'said', 'range', 'never', 'available', 'took', 'later', 'even', 'brilliant', 'wanted', 'next', 'communication', 'found']
```

- Looking at the top 100 most frequent words, we can get some idea about the dataset
- single words do not have context and hence we try to look at top 50 bigrams/trigrams

## Top 50 bigrams

```bash
[('good', 'price'), ('great', 'service'), ('easy', 'use'), ('excellent', 'service'), ('great', 'price'), ('tyre', 'fitted'), ('good', 'service'), ('local', 'garage'), ('customer', 'service'), ('competitive', 'price'), ('new', 'tyre'), ('price', 'good'), ('good', 'value'), ('best', 'price'), ('easy', 'order'), ('definitely', 'use'), ('service', 'good'), ('quick', 'easy'), ('fitting', 'centre'), ('service', 'great'), ('choice', 'tyre'), ('value', 'money'), ('tyre', 'fitting'), ('efficient', 'service'), ('price', 'easy'), ('price', 'tyre'), ('tyre', 'good'), ('tyre', 'delivered'), ('tyre', 'fitter'), ('use', 'website'), ('service', 'easy'), ('fitting', 'service'), ('good', 'choice'), ('hassle', 'free'), ('great', 'value'), ('excellent', 'price'), ('price', 'great'), ('get', 'tyre'), ('fit', 'tyre'), ('price', 'service'), ('highly', 'recommend'), ('time', 'used'), ('start', 'finish'), ('first', 'class'), ('first', 'time'), ('easy', 'book'), ('fitted', 'tyre'), ('fitting', 'garage'), ('garage', 'fitted'), ('reasonable', 'price'), ('quality', 'tyre')]
```

## Top 50 Trigrams

```bash
[('easy', 'use', 'website'), ('service', 'good', 'price'), ('service', 'great', 'price'), ('tyre', 'good', 'price'), ('good', 'price', 'good'), ('service', 'easy', 'use'), ('good', 'price', 'easy'), ('price', 'great', 'service'), ('great', 'service', 'great'), ('good', 'price', 'tyre'), ('website', 'easy', 'use'), ('great', 'price', 'great'), ('price', 'excellent', 'service'), ('price', 'good', 'service'), ('good', 'choice', 'tyre'), ('first', 'class', 'service'), ('good', 'value', 'money'), ('easy', 'use', 'great'), ('price', 'easy', 'use'), ('great', 'service', 'easy'), ('great', 'price', 'easy'), ('good', 'service', 'good'), ('tyre', 'great', 'price'), ('service', 'competitive', 'price'), ('get', 'tyre', 'fitted'), ('tyre', 'competitive', 'price'), ('good', 'selection', 'tyre'), ('way', 'buy', 'tyre'), ('garage', 'fitted', 'tyre'), ('fitted', 'local', 'garage'), ('new', 'tyre', 'fitted'), ('great', 'service', 'good'), ('easy', 'use', 'good'), ('second', 'time', 'used'), ('excellent', 'customer', 'service'), ('great', 'customer', 'service'), ('great', 'choice', 'tyre'), ('service', 'start', 'finish'), ('good', 'price', 'excellent'), ('great', 'value', 'money'), ('first', 'time', 'used'), ('excellent', 'service', 'easy'), ('tyre', 'fitted', 'local'), ('use', 'website', 'good'), ('good', 'price', 'service'), ('local', 'garage', 'fit'), ('excellent', 'service', 'good'), ('first', 'time', 'using'), ('go', 'anywhere', 'else'), ('excellent', 'price', 'service'), ('good', 'quality', 'tyre')]
```

## Insights from bigram trigram analysis
- We know that unigrams don't tell us much about context
- bigrams/trigrams have some context with them. These are mainly some adj/adv combined with noun/verb which gives us more information. 
- Hence, involving bigrams/trigrams in our topic clusters make more sense.

## Labelling the documents with the relevant Topics

Use CreateDataForClassification.ipynb notebook for labelling the documents with the relevant topics

- We check the presence of cluster keywords in the documents. 
- If more than 3 words are present in the document, then we assign that cluster name as relevant topic (label) to that document

![Dataset snap shot after labelling](LabelledDocsDatasetSnap.png?raw=true "Dataset snap shot after labelling")

## Relevant Topic Distribution

![Relevant topic distribution after labelling](RelevantTopicDistribution.png?raw=true "Relevant topic distribution after labelling")
