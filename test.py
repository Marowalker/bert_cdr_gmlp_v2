from dataset import parse_words, Dataset
import constants
from data_utils import make_triple_vocab, load_vocab, get_trimmed_w2v_vectors
import pickle
import tensorflow as tf


vocab_poses = load_vocab(constants.ALL_POSES)
vocab_synsets = load_vocab(constants.ALL_SYNSETS)
vocab_rels = load_vocab(constants.ALL_DEPENDS)

# with open(constants.RAW_DATA + 'sdp_data_acentors_bert.train.txt') as f:
#     lines = f.readlines()
#
# words, poses, synsets, relations, labels, identities, triples = parse_words(lines)
#
# print(relations)

chem_vocab = make_triple_vocab(constants.DATA + 'chemical2id.txt')
dis_vocab = make_triple_vocab(constants.DATA + 'disease2id.txt')

train = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.train.txt', vocab_poses=vocab_poses,
                vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
dev = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.dev.txt', vocab_poses=vocab_poses,
              vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)
test = Dataset(constants.RAW_DATA + 'sdp_data_acentors_bert.test.txt', vocab_poses=vocab_poses,
               vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab, vocab_dis=dis_vocab)

# Train, Validation Split
validation = Dataset('', '', process_data=False)
train_ratio = 0.85
n_sample = int(len(dev.words) * (2 * train_ratio - 1))
props = ['words', 'head_mask', 'e1_mask', 'e2_mask', 'labels', 'poses', 'synsets', 'identities', 'triples']

for prop in props:
    train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
    validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

train.get_padded_data()
validation.get_padded_data()

print(len(train.words))

# wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')
#
# # print(wn_emb.shape)
#
# with open(constants.EMBEDDING_CHEM, 'rb') as f:
#     chem_emb = pickle.load(f)
#     f.close()
#
# with open(constants.EMBEDDING_DIS, 'rb') as f:
#     dis_emb = pickle.load(f)
#     f.close()
#
# concated = tf.concat([chem_emb, dis_emb], axis=0)
# print(concated)
