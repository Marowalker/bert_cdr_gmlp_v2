from dataset import Dataset
import constants
from data_utils import make_triple_vocab, load_vocab, get_trimmed_w2v_vectors
import pickle
import tensorflow as tf
from evaluate.bc5 import evaluate_bc5
from gmlp.model.bert_gmlp import BertgMLPModel
from gmlp.model.bert_cnn import BertCNNModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np


# from gmlp.model.bert_gmlp_compat import BERTgMLPModel


def main(mode='cid'):
    if mode == 'cid':
        if constants.IS_REBUILD == 1:
            print('Build data')
            vocab_words = load_vocab(constants.ALL_WORDS)
            vocab_poses = load_vocab(constants.ALL_POSES)
            vocab_synsets = load_vocab(constants.ALL_SYNSETS)
            vocab_rels = load_vocab(constants.ALL_DEPENDS)
            chem_vocab = make_triple_vocab(constants.DATA + 'chemical2id.txt')
            dis_vocab = make_triple_vocab(constants.DATA + 'disease2id.txt')

            train = Dataset(constants.RAW_DATA + 'sentence_data_acentors.train.txt',
                            constants.RAW_DATA + 'sdp_data_acentors_bert.train.txt',
                            vocab_words=vocab_words,
                            vocab_poses=vocab_poses,
                            vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                            vocab_dis=dis_vocab,
                            process_data='cid')
            pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

            dev = Dataset(constants.RAW_DATA + 'sentence_data_acentors.dev.txt',
                          constants.RAW_DATA + 'sdp_data_acentors_bert.dev.txt',
                          vocab_words=vocab_words,
                          vocab_poses=vocab_poses,
                          vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                          vocab_dis=dis_vocab,
                          process_data='cid')
            pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

            test = Dataset(constants.RAW_DATA + 'sentence_data_acentors.test.txt',
                           constants.RAW_DATA + 'sdp_data_acentors_bert.test.txt',
                           vocab_words=vocab_words,
                           vocab_poses=vocab_poses,
                           vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                           vocab_dis=dis_vocab,
                           process_data='cid')
            pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

        else:
            print('Load data')
            train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
            dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
            test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

        with open('data/w2v_model/transe_cdr_word_16.pkl', 'rb') as f:
            word_emb = pickle.load(f)
            f.close()

        with open('data/w2v_model/transe_cdr_relation_16.pkl', 'rb') as f:
            rel_emb = pickle.load(f)
            f.close()
    else:
        if constants.IS_REBUILD == 1:
            print('Build data')
            vocab_words = load_vocab(constants.ALL_WORDS)
            vocab_poses = load_vocab(constants.ALL_POSES)
            vocab_synsets = load_vocab(constants.ALL_SYNSETS)
            vocab_rels = load_vocab(constants.ALL_DEPENDS)
            chem_vocab = make_triple_vocab(constants.DATA + 'chemprot_chemical2id.txt')
            dis_vocab = make_triple_vocab(constants.DATA + 'chemprot_gene2id.txt')

            train = Dataset(constants.CHEMPROT_DATA + 'sentence_data_acentors_no_kb.train.txt',
                            constants.CHEMPROT_DATA + 'sdp_data_acentors_no_kb.train.txt',
                            vocab_words=vocab_words,
                            vocab_poses=vocab_poses,
                            vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                            vocab_dis=dis_vocab,
                            process_data='chemprot')
            pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
            #
            dev = Dataset(constants.CHEMPROT_DATA + 'sentence_data_acentors_no_kb.dev.txt',
                          constants.CHEMPROT_DATA + 'sdp_data_acentors_no_kb.dev.txt',
                          vocab_words=vocab_words,
                          vocab_poses=vocab_poses,
                          vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                          vocab_dis=dis_vocab,
                          process_data='chemprot')
            pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

            test = Dataset(constants.CHEMPROT_DATA + 'sentence_data_acentors_no_kb.test.txt',
                           constants.CHEMPROT_DATA + 'sdp_data_acentors_no_kb.test.txt',
                           vocab_words=vocab_words,
                           vocab_poses=vocab_poses,
                           vocab_synset=vocab_synsets, vocab_rels=vocab_rels, vocab_chems=chem_vocab,
                           vocab_dis=dis_vocab,
                           process_data='chemprot')
            pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

        else:
            print('Load data')
            train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
            dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
            test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

        with open('data/w2v_model/transe_chemprot_word_16.pkl', 'rb') as f:
            word_emb = pickle.load(f)
            f.close()

        with open('data/w2v_model/transe_chemprot_relation_16.pkl', 'rb') as f:
            rel_emb = pickle.load(f)
            f.close()

    # Train, Validation Split
    validation = Dataset('', '', process_data=None)
    train_ratio = 0.85
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'head_mask', 'e1_mask', 'e2_mask', 'labels', 'poses', 'synsets', 'relations', 'identities',
             'triples']
    # props = ['words', 'head_mask', 'e1_mask', 'e2_mask', 'labels', 'poses', 'synsets', 'identities',
    #          'triples']

    for prop in props:
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    train.get_padded_data(mode='chemprot')
    validation.get_padded_data(mode='chemprot')
    test.get_padded_data(mode='chemprot', shuffled=False)

    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    wn_emb = get_trimmed_w2v_vectors('data/w2v_model/wordnet_embeddings.npz')

    with open(constants.EMBEDDING_CHEM, 'rb') as f:
        chem_emb = pickle.load(f)
        f.close()

    with open(constants.EMBEDDING_DIS, 'rb') as f:
        dis_emb = pickle.load(f)
        f.close()

    with tf.device("/GPU:0"):
        model = BertgMLPModel(base_encoder=constants.encoder, depth=6, chem_emb=chem_emb, dis_emb=dis_emb,
                              wordnet_emb=wn_emb, cdr_emb=word_emb, rel_emb=rel_emb, mode=mode)
        # model = BertCNNModel(base_encoder=constants.encoder, chem_emb=chem_emb, dis_emb=dis_emb,
        #                      wordnet_emb=wn_emb, cdr_emb=word_emb, rel_emb=rel_emb)
        model.build(train, validation)

        y_pred = model.predict(test)

        answer = {}
        identities = test.identities

    for i in range(len(y_pred)):
        if mode == 'cid':
            if y_pred[i] == 0:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])
        else:
            if y_pred[i] != 5:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])
    print(
        'result: abstract: ', evaluate_bc5(answer)
    )


if __name__ == '__main__':
    main(mode='chemprot')