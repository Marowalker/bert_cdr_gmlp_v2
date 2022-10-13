import argparse
from transformers import TFBertModel, BertTokenizer
from data_utils import load_vocab

ALL_LABELS = ['CID', 'NONE']

UNK = '$UNK$'

parser = argparse.ArgumentParser(description='Multi-region size gMLP with BERT for re')
parser.add_argument('-i', help='Job identity', type=int, default=0)
parser.add_argument('-rb', help='Rebuild data', type=int, default=1)
parser.add_argument('-e', help='Number of epochs', type=int, default=20)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=5)
parser.add_argument('-config', help='CNN configurations default \'1:128\'', type=str, default='2:32')
parser.add_argument('-len', help='Max sentence or document length', type=int, default=255)


opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i
IS_REBUILD = opt.rb
EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p
DROPOUT = 0.1

# INPUT_W2V_DIM = 300
# INPUT_W2V_DIM = 200
INPUT_W2V_DIM = 768
TRIPLE_W2V_DIM = 200

MAX_LENGTH = opt.len

CNN_FILTERS = {}
if opt.config:
    print('Use model CNN with config', opt.config)
    USE_CNN = True
    CNN_FILTERS = {
        int(k): int(f) for k, f in [i.split(':') for i in opt.config.split(',')]
    }
else:
    raise ValueError('Configure CNN model to start')


DATA = 'data/'
RAW_DATA = DATA + 'raw_data/'
PICKLE_DATA = DATA + 'pickle/'
W2V_DATA = DATA + 'w2v_model/'

EMBEDDING_CHEM = W2V_DATA + 'transh_chemical_embeddings_200.pkl'
EMBEDDING_DIS = W2V_DATA + 'transh_disease_embeddings_200.pkl'

# ALL_WORDS = DATA + 'vocab.txt'
ALL_WORDS = DATA + 'all_words.txt'
ALL_POSES = DATA + 'all_pos.txt'
ALL_SYNSETS = DATA + 'all_hypernyms.txt'
# ALL_SYNSETS = DATA + 'all_synsets.txt'
# ALL_DEPENDS = DATA + 'all_depend.txt'
ALL_DEPENDS = DATA + 'no_dir_depend.txt'

# encoder = TFBertModel.from_pretrained("dmis-lab/biobert-v1.1", from_pt=True)
# tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
encoder = TFBertModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", from_pt=True)
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
# vocab_depend = load_vocab(ALL_DEPENDS)
#
# for d in vocab_depend:
#     if d != '':
#         ADDITIONAL_SPECIAL_TOKENS.append(d.strip())

# print(ADDITIONAL_SPECIAL_TOKENS)

tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

START_E1 = tokenizer.encode('<e1>')[1]
END_E1 = tokenizer.encode('</e1>')[1]
START_E2 = tokenizer.encode('<e2>')[1]
END_E2 = tokenizer.encode('</e2>')[1]

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
