import tensorflow as tf
import constants
from constants import *
import numpy as np
import os
import keras.backend as K
from data_utils import *


def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


class BertgMLPModel:
    def __init__(self, base_encoder, depth, chem_emb, dis_emb, wordnet_emb, cdr_emb, rel_emb):
        if not os.path.exists(TRAINED_MODELS):
            os.makedirs(TRAINED_MODELS)

        self.encoder = base_encoder
        self.depth = depth
        self.triple_emb = tf.concat([chem_emb, dis_emb], axis=0)
        self.wordnet_emb = wordnet_emb
        self.cdr_emb = tf.concat([cdr_emb, rel_emb], axis=0)

        self.max_length = constants.MAX_LENGTH
        self.num_of_words = countVocab()
        self.num_of_pos = countNumPos()
        self.num_of_synset = countNumSynset()
        self.num_of_depend = countNumRelation()
        self.num_of_class = len(constants.ALL_LABELS)
        self.trained_models = constants.TRAINED_MODELS

    def _add_inputs(self):
        self.input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.head_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype='float32')
        self.e1_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype='float32')
        self.e2_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype='float32')
        self.pos_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.synset_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype='int32')
        self.relation_ids = tf.keras.layers.Input(shape=(36,), dtype='int32')
        self.triple_ids = tf.keras.layers.Input(shape=(2,), dtype='int32')

    def _bert_layer(self):
        self.bertoutput = self.encoder(self.input_ids)
        emb = self.bertoutput[0]

        pos_emb = tf.keras.layers.Embedding(self.num_of_pos + 1, 6)(self.pos_ids)

        synset_emb = tf.keras.layers.Embedding(self.wordnet_emb.shape[0], 18, weights=[self.wordnet_emb],
                                               trainable=False)(self.synset_ids)

        triple_emb = tf.keras.layers.Embedding(self.triple_emb.shape[0], constants.TRIPLE_W2V_DIM,
                                               weights=[self.triple_emb], trainable=False)(self.triple_ids)

        relation_emb = tf.keras.layers.Embedding(self.num_of_words + self.num_of_depend + 2, 16,
                                                 weights=[self.cdr_emb], trainable=False)(self.relation_ids)

        word_x = emb
        head_x = mat_mul(word_x, self.head_mask)
        head_x = tf.keras.layers.Dropout(constants.DROPOUT)(head_x)
        head_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(head_x)

        e1_x = mat_mul(word_x, self.e1_mask)
        e1_x = tf.keras.layers.Dropout(constants.DROPOUT)(e1_x)
        e1_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(e1_x)

        e2_x = mat_mul(word_x, self.e2_mask)
        e2_x = tf.keras.layers.Dropout(constants.DROPOUT)(e2_x)
        e2_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(e2_x)

        pos_emb = tf.expand_dims(pos_emb, -1)
        synset_emb = tf.expand_dims(synset_emb, -1)
        triple_emb = tf.expand_dims(triple_emb, -1)
        relation_emb = tf.expand_dims(relation_emb, -1)

        cnn_outputs = []
        for k in constants.CNN_FILTERS:
            filters = constants.CNN_FILTERS[k]

            pos_x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 6),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(pos_emb)

            synset_x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 18),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(synset_emb)

            triple_x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, constants.TRIPLE_W2V_DIM),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(triple_emb)

            relation_x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(k, 16),
                strides=(1, 1),
                activation='tanh',
                use_bias=False, padding="valid",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(relation_emb)

            cnn_output = tf.concat(
                [pos_x, synset_x, triple_x, relation_x],
                axis=1)
            cnn_output = tf.reduce_max(input_tensor=cnn_output, axis=1)
            cnn_output = tf.reshape(cnn_output, [-1, filters])
            cnn_outputs.append(cnn_output)

        final_cnn_output = tf.concat(cnn_outputs, axis=-1)
        final_cnn_output = tf.nn.dropout(final_cnn_output, DROPOUT)

        x = tf.keras.layers.concatenate([head_x, e1_x, e2_x, final_cnn_output])

        out = tf.keras.layers.Dropout(DROPOUT)(x)
        out = tf.keras.layers.Dense(128)(out)
        out = tf.keras.layers.Dense(128)(out)
        out = tf.keras.layers.Dense(len(constants.ALL_LABELS), activation='softmax')(out)
        return out

    @staticmethod
    def f1_score(y_true, y_pred):
        return f1_macro(y_true, y_pred)

    def _add_train_ops(self):
        self.model = tf.keras.Model(
            inputs=[self.input_ids, self.head_mask, self.e1_mask, self.e2_mask, self.pos_ids, self.synset_ids,
                    self.relation_ids, self.triple_ids],
            outputs=self._bert_layer())
        self.optimizer = tf.keras.optimizers.Adam(lr=4e-6)

        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy', self.f1_score])
        print(self.model.summary())

    def _train(self, train_data, val_data):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max',
                                                          patience=constants.PATIENCE)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=TRAINED_MODELS,
            save_weights_only=True,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True)

        self.model.fit(x=(train_data.words, train_data.head_mask, train_data.e1_mask, train_data.e2_mask,
                          train_data.poses, train_data.synsets, train_data.relations, train_data.triples),
                       y=train_data.labels,
                       validation_data=((val_data.words, val_data.head_mask, val_data.e1_mask, val_data.e2_mask,
                                         val_data.poses, val_data.synsets, val_data.relations, val_data.triples),
                                        val_data.labels),
                       batch_size=16, epochs=constants.EPOCHS, callbacks=[early_stopping, model_checkpoint_callback])

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_layer_names=True,
                                  rankdir='TB',
                                  expand_nested=False, dpi=300)

    def build(self, train_data, val_data):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            self._train(train_data, val_data)
            self.plot_model()

    def predict(self, test_data):
        self.model.load_weights(TRAINED_MODELS)
        pred = self.model.predict([test_data.words, test_data.head_mask, test_data.e1_mask, test_data.e2_mask,
                                   test_data.poses, test_data.synsets, test_data.relations, test_data.triples])
        y_pred = []
        for logit in pred:
            y_pred.append(np.argmax(logit))
        return y_pred
