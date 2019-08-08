# The sentence objects holds a sentence that we may want to embed or tag
import sys
#sys.path.insert(0, '/Users/buyukozb/git/thesis/mylibs')
from flair.embeddings import *
from flair.data_fetcher import *
from pathlib import Path
#BERT_BASE_DIR="/Users/buyukozb/Desktop/berfu/thesis/data/embedding_data/uncased_L-12_H-768_A-12"
# init embedding
#embedding = BertEmbeddings(BERT_BASE_DIR)

'''
# create a sentence
sentence = Sentence('The grass is green .')
# embed words in sentence
embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
'''

# initialize the document embeddings, mode = mean
#document_embeddings = DocumentPoolEmbeddings([embedding])

# create an example sentence
#sentence = Sentence('The grass is green . And the sky is blue .')

# embed the sentence with our document embedding
#document_embeddings.embed(sentence)

# now check out the embedded sentence.
#print(sentence.get_embedding())

#####
#from pytorch_pretrained_bert import BertTokenizer
# berttokenizer does basic tokenization before wordpiece as default behavior.
# basic tokenization means: punctuation splitting, lower casing, etc.
# data argument must be raw text.
#tokenizer = BertTokenizer.from_pretrained(BERT_BASE_DIR)
#tokens = tokenizer.tokenize('The grass is green . And the sky is blue .')

# use your own data path
data_folder = Path('/Users/buyukozb/git/berfu/thesis/data/all_data/india/flair_formatted')

# load corpus containing training, test and dev data
# in load_classification_corpus, use_tokenizer = True as default. It uses segtok tokenizer. Change it to BertTokenizer.
# Differences from bert_tpu model:
# in original data (rcv1 china), sentences are separated with \n. In the flair version of the data, \n are removed. Sentences in a document are separated with simply whitespace.
# This arrangement was needed because flair wants data in a format that on each document is on a single line.
# Data is splitted into train/test by using Sklearn. But here, if there is no dev set available, flair takes 0.1 of the train as dev set.
'''
corpus = NLPTaskDataFetcher.load_classification_corpus(data_folder,test_file='test.txt',train_file='train.txt',
                                                       tokenizer_name='bert',
                                                       bert_model_or_path=BERT_BASE_DIR)
'''

#by default takes documents as a whole unless max_tokens_per_doc is specified.
corpus = NLPTaskDataFetcher.load_classification_corpus(data_folder,train_folder_name='train',test_folder_name='test', dev_split_size=0, max_seq_len=128)

from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
#embedding = BertEmbeddings(BERT_BASE_DIR)
embedding = ELMoEmbeddings(model="small")

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
# sitepackages/torch/nn/modules/module 489 had a linebreak but I removed it since it was looping over it.
# mean pooling.
document_embeddings = DocumentPoolEmbeddings([embedding])

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
# Training aborted due to excessive size of documents. With each document limited to 5 sentences, training succesfully performed.
# But the main reason I tried this tool was to overcome maximum length imposed in BERT.
# So a workaround will not be helpful.
trainer.train(data_folder,
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=2,
              embeddings_in_memory=False,
              evaluation_metric=EvaluationMetric.MACRO_F1_SCORE) #ilk ornekte True idi. False yapinca da bir sey degismedi sorunu cozmede.

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves(data_folder + '/loss.tsv')
plotter.plot_weights(data_folder + '/weights.txt')