import numpy as np
from nltk.corpus import wordnet as wn
import constants
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.random.seed(13)


def parse_words(raw_data):
    all_words = []
    all_poses = []
    all_synsets = []
    all_labels = []
    all_identities = []
    all_triples = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
        else:
            pair = l[0]
            label = l[1]
            if label:
                chem, dis = pair.split('_')
                all_triples.append([chem, dis])

                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    poses = []
                    synsets = []
                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        if idx % 2 == 0:
                            for idx, _node in enumerate(node):
                                word = constants.UNK if _node == '' else _node
                                if idx == 0:
                                    w, p, s = word.split('\\')
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('entity')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    words.append(_w)
                                    poses.append(p)
                                    synsets.append(s)
                                else:
                                    w = word.split('\\')[0]
                        else:
                            pass

                    all_words.append(words)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                print(l)

    return all_words, all_poses, all_synsets, all_labels, all_identities, all_triples


class Dataset:
    def __init__(self, data_name, vocab_poses=None, vocab_synset=None, vocab_chems=None, vocab_dis=None,
                 process_data=True):
        self.data_name = data_name

        self.labels = None
        self.poses = None
        self.synsets = None
        self.identities = None

        self.vocab_poses = vocab_poses
        self.vocab_synsets = vocab_synset

        self.vocab_chems = vocab_chems
        self.vocab_dis = vocab_dis

        if process_data:
            self._process_data()
            # self._pad_data()
            self._clean_data()

    def get_padded_data(self, shuffled=True):
        self._pad_data(shuffled=shuffled)

    def _clean_data(self):
        del self.vocab_poses
        del self.vocab_synsets

    def _process_data(self):
        with open(self.data_name, 'r') as f:
            raw_data = f.readlines()
        data_words, data_pos, data_synsets, data_y, self.identities, data_triples = parse_words(raw_data)

        words = []
        labels = []
        poses = []
        synsets = []

        for tokens in data_words:
            sdp_sent = ' '.join(tokens)
            token_ids = constants.tokenizer.encode(sdp_sent)
            words.append(token_ids)

        for i in range(len(data_pos)):

            ps, ss = [], []

            for p, s in zip(data_pos[i], data_synsets[i]):
                if p in self.vocab_poses:
                    p_id = self.vocab_poses[p]
                else:
                    p_id = self.vocab_poses['NN']
                ps += [p_id]
                if s in self.vocab_synsets:
                    synset_id = self.vocab_synsets[s]
                else:
                    synset_id = self.vocab_synsets[constants.UNK]
                ss += [synset_id]

            poses.append(ps)
            synsets.append(ss)

            lb = constants.ALL_LABELS.index(data_y[i][0])
            # if data_y[i][0] == 'CID':
            #     lb = [1, 0]
            # else:
            #     lb = [0, 1]
            labels.append(lb)

        self.words = words
        self.labels = labels
        self.poses = poses
        self.synsets = synsets
        self.triples = self.parse_triple(data_triples)

    def parse_triple(self, all_triples):
        data_triples = []
        for c, d in all_triples:
            c_id = int(self.vocab_chems[c])
            d_id = int(self.vocab_dis[d]) + int(len(self.vocab_chems))
            data_triples.append([c_id, d_id])

        return data_triples

    def _pad_data(self, shuffled=True):
        if shuffled:
            word_shuffled, pos_shuffled, synset_shuffled, label_shuffled, triple_shuffled = shuffle(
                self.words, self.poses, self.synsets, self.labels, self.triples
            )
        else:
            word_shuffled, pos_shuffled, synset_shuffled, label_shuffled, triple_shuffled = self.words, self.poses, \
                                                                                            self.synsets, self.labels,\
                                                                                            self.triples

        self.words = tf.constant(pad_sequences(word_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.poses = tf.constant(pad_sequences(pos_shuffled,  maxlen=constants.MAX_LENGTH, padding='post'))
        self.synsets = tf.constant(pad_sequences(synset_shuffled, maxlen=constants.MAX_LENGTH, padding='post'))
        self.labels = tf.keras.utils.to_categorical(label_shuffled)
        self.triples = tf.constant(triple_shuffled, dtype='int32')

