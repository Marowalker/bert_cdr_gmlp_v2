# import os
# import numpy as np
# import tensorflow as tf
# from utils import Timer, Log
# from data_utils import countNumPos, countNumSynset, countVocab, mat_mul
# from sklearn.utils import shuffle
# import constants
# from sklearn.metrics import f1_score
# from gmlp.gmlp import gMLP
# from dataset import my_pad_sequences
#
# tf.random.Generator = None
#
# seed = 1234
# np.random.seed(seed)
#
#
# # tf.compat.v1.disable_eager_execution()
# # tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
#
#
# class BERTgMLPModel:
#     def __init__(self, model_name, base_encoder, depth, chem_emb, dis_emb, wordnet_emb):
#         self.model_name = model_name
#         self.embeddings = base_encoder
#         self.triples = tf.concat([chem_emb, dis_emb], axis=0)
#         self.batch_size = 16
#         self.wordnet_emb = wordnet_emb
#         self.depth = depth
#
#         self.max_length = constants.MAX_LENGTH
#         # Num of pos tags
#         self.num_of_pos = countNumPos()
#         self.num_of_synset = countNumSynset()
#         self.num_of_siblings = countVocab()
#         self.num_of_class = len(constants.ALL_LABELS)
#         self.trained_models = constants.TRAINED_MODELS
#         self.initializer = tf.initializers.glorot_normal()
#
#     def _add_placeholders(self):
#         """
#         Adds placeholders to self
#         """
#         self.labels = tf.compat.v1.placeholder(shape=[None, self.num_of_class], dtype='int32')
#         # Indexes of first channel (word + dependency relations)
#         self.word_ids = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='int32')
#         # Indexes of second channel (pos tags + dependency relations)
#         self.pos_ids = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='int32')
#         # Indexes of fourth channel (synset + dependency relations)
#         self.synset_ids = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='int32')
#
#         self.triple_ids = tf.compat.v1.placeholder(shape=[None, 2], dtype='int32')
#
#         self.head_mask = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='float32')
#         self.e1_mask = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='float32')
#         self.e2_mask = tf.compat.v1.placeholder(shape=[None, self.max_length], dtype='float32')
#
#         # self.dropout_embedding = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout_embedding")
#         # self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="dropout")
#         # self.is_training = tf.compat.v1.placeholder(tf.bool, name='phase')
#
#     def _add_word_embeddings_op(self):
#         """
#         Adds word embeddings to self
#         """
#         self.bertoutput = self.embeddings(self.word_ids)
#         self.word_embeddings = self.bertoutput[0]
#
#         self.pos_embeddings = tf.keras.layers.Embedding(self.num_of_pos + 1, 6)(self.pos_ids)
#
#         self.synset_embeddings = tf.keras.layers.Embedding(self.wordnet_emb.shape[0], 18, weights=[self.wordnet_emb],
#                                                            trainable=False)(self.synset_ids)
#
#         self.triple_embeddings = tf.keras.layers.Embedding(self.triples.shape[0], constants.TRIPLE_W2V_DIM,
#                                                            weights=[self.triples], trainable=False)(self.triple_ids)
#
#     def _multiple_input_gmlp_layers(self):
#         word_x = gMLP(dim=constants.INPUT_W2V_DIM, depth=self.depth, seq_len=self.max_length,
#                       activation=tf.nn.swish)(self.word_embeddings)
#         pos_x = gMLP(dim=6, depth=self.depth, seq_len=self.max_length, activation=tf.nn.swish)(self.pos_embeddings)
#         synset_x = gMLP(dim=18, depth=self.depth, seq_len=self.max_length, activation=tf.nn.swish) \
#             (self.synset_embeddings)
#         triple_x = gMLP(dim=constants.TRIPLE_W2V_DIM, depth=self.depth, seq_len=2, activation=tf.nn.swish) \
#             (self.triple_embeddings)
#
#         head_x = mat_mul(word_x, self.head_mask)
#         head_x = tf.keras.layers.Dropout(constants.DROPOUT)(head_x)
#         head_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(head_x)
#
#         e1_x = mat_mul(word_x, self.e1_mask)
#         e1_x = tf.keras.layers.Dropout(constants.DROPOUT)(e1_x)
#         e1_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(e1_x)
#
#         e2_x = mat_mul(word_x, self.e2_mask)
#         e2_x = tf.keras.layers.Dropout(constants.DROPOUT)(e2_x)
#         e2_x = tf.keras.layers.Dense(constants.INPUT_W2V_DIM)(e2_x)
#
#         pos_x = tf.keras.layers.Flatten(data_format="channels_first")(pos_x)
#         pos_x = tf.keras.layers.LayerNormalization()(pos_x)
#         pos_x = tf.keras.layers.Dropout(constants.DROPOUT)(pos_x)
#         pos_x = tf.keras.layers.Dense(6)(pos_x)
#
#         synset_x = tf.keras.layers.Flatten(data_format="channels_first")(synset_x)
#         synset_x = tf.keras.layers.LayerNormalization()(synset_x)
#         synset_x = tf.keras.layers.Dropout(constants.DROPOUT)(synset_x)
#         synset_x = tf.keras.layers.Dense(18)(synset_x)
#
#         triple_x = tf.keras.layers.Flatten(data_format="channels_first")(triple_x)
#         triple_x = tf.keras.layers.LayerNormalization()(triple_x)
#         triple_x = tf.keras.layers.Dropout(constants.DROPOUT)(triple_x)
#         triple_x = tf.keras.layers.Dense(constants.TRIPLE_W2V_DIM)(triple_x)
#
#         x = tf.keras.layers.concatenate([head_x, e1_x, e2_x, pos_x, synset_x, triple_x])
#
#         return x
#
#     def _add_logits_op(self):
#         """
#         Adds logits to self
#         """
#         final_cnn_output = self._multiple_input_gmlp_layers()
#         hidden_1 = tf.keras.layers.Dense(
#             units=128, name="hidden_1",
#             kernel_initializer=tf.keras.initializers.GlorotNormal(),
#             kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#         )(final_cnn_output)
#         hidden_2 = tf.keras.layers.Dense(
#             units=128, name="hidden_2",
#             kernel_initializer=tf.keras.initializers.GlorotNormal(),
#             kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#         )(hidden_1)
#         self.outputs = tf.keras.layers.Dense(
#             units=self.num_of_class,
#             activation=tf.nn.softmax,
#             kernel_initializer=tf.keras.initializers.GlorotNormal(),
#             kernel_regularizer=tf.keras.regularizers.l2(1e-4)
#         )(hidden_2)
#         self.logits = tf.nn.softmax(self.outputs)
#
#     def _add_loss_op(self):
#         with tf.compat.v1.variable_scope('loss_layers'):
#             log_likelihood = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
#             regularizer = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
#             self.loss = tf.reduce_mean(input_tensor=log_likelihood)
#             self.loss += tf.reduce_sum(input_tensor=regularizer)
#
#     def _add_train_op(self):
#         self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
#
#         with tf.compat.v1.variable_scope("train_step"):
#             tvars = tf.compat.v1.trainable_variables()
#             grad, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=tvars), 100.0)
#             optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
#             self.train_op = optimizer.apply_gradients(zip(grad, tvars))
#
#     def build(self):
#         timer = Timer()
#         timer.start("Building model...")
#
#         self._add_placeholders()
#         self._add_word_embeddings_op()
#         self._add_logits_op()
#         self._add_loss_op()
#         self._add_train_op()
#
#         timer.stop()
#
#     def _next_batch(self, data, num_batch):
#         start = 0
#         idx = 0
#         while idx < num_batch:
#             # Get BATCH_SIZE samples each batch
#             word_ids = data['words'][start:start + self.batch_size]
#             pos_ids = data['poses'][start:start + self.batch_size]
#             synset_ids = data['synsets'][start:start + self.batch_size]
#
#             head_mask = data['head_mask'][start: start + self.batch_size]
#             e1_mask = data['e1_mask'][start: start + self.batch_size]
#             e2_mask = data['e2_mask'][start: start + self.batch_size]
#             labels = data['labels'][start:start + self.batch_size]
#             triple_ids = data['triples'][start: start + self.batch_size]
#
#             word_ids, _ = my_pad_sequences(word_ids, pad_tok=0, max_sent_length=self.max_length)
#             pos_ids, _ = my_pad_sequences(pos_ids, pad_tok=0, max_sent_length=self.max_length)
#             synset_ids, _ = my_pad_sequences(synset_ids, pad_tok=0, max_sent_length=self.max_length)
#             labels = tf.keras.utils.to_categorical(labels)
#
#             head_mask = np.array(head_mask)
#             e1_mask = np.array(e1_mask)
#             e2_mask = np.array(e2_mask)
#             triple_ids = np.array(triple_ids)
#
#             start += self.batch_size
#             idx += 1
#             yield word_ids, pos_ids, synset_ids, triple_ids, head_mask, e1_mask, e2_mask, labels
#
#     def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
#         Log.verbose = verbose
#         if not os.path.exists(self.trained_models):
#             os.makedirs(self.trained_models)
#
#         saver = tf.compat.v1.train.Saver(max_to_keep=2)
#         best_f1 = 0
#         n_epoch_no_improvement = 0
#         with tf.compat.v1.Session() as sess:
#             sess.run(tf.compat.v1.global_variables_initializer())
#             num_batch_train = len(self.dataset_train.labels) // self.batch_size + 1
#             for e in range(epochs):
#                 # print(len(self.dataset_train.siblings))
#
#                 words_shuffled, poses_shuffled, synsets_shuffled, triples_shuffled, head_shuffled, e1_shuffled, \
#                 e2_shuffled, labels_shuffled = shuffle(
#                     self.dataset_train.words,
#                     self.dataset_train.poses,
#                     self.dataset_train.synsets,
#                     self.dataset_train.triples,
#                     self.dataset_train.head_mask,
#                     self.dataset_train.e1_mask,
#                     self.dataset_train.e2_mask,
#                     self.dataset_train.labels
#                 )
#
#                 data = {
#                     'words': words_shuffled,
#                     'poses': poses_shuffled,
#                     'synsets': synsets_shuffled,
#                     'triples': triples_shuffled,
#                     'head_mask': head_shuffled,
#                     'e1_mask': e1_shuffled,
#                     'e2_mask': e2_shuffled,
#                     'labels': labels_shuffled,
#                 }
#
#                 for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_train)):
#                     word_ids, pos_ids, synset_ids, triple_ids, head_mask, e1_mask, e2_mask, labels \
#                         = batch
#                     # positions_1, positions_2, word_ids, pos_ids, synset_ids, relation_ids, labels = batch
#                     feed_dict = {
#                         self.word_ids: word_ids,
#                         self.pos_ids: pos_ids,
#                         self.synset_ids: synset_ids,
#                         self.labels: labels,
#                         self.triple_ids: triple_ids,
#                         self.head_mask: head_mask,
#                         self.e1_mask: e1_mask,
#                         self.e2_mask: e2_mask
#                     }
#                     _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
#                     if idx % 10 == 0:
#                         Log.log("Iter {}, Loss: {} ".format(idx, loss_train))
#
#                 # stop by validation loss
#                 if early_stopping:
#                     num_batch_val = len(self.dataset_validation.labels) // self.batch_size + 1
#                     total_f1 = []
#
#                     data = {
#                         'words': self.dataset_validation.words,
#                         'poses': self.dataset_validation.poses,
#                         'synsets': self.dataset_validation.synsets,
#                         'labels': self.dataset_validation.labels,
#                         'triples': self.dataset_validation.triples,
#                         'head_mask': self.dataset_validation.head_mask,
#                         'e1_mask': self.dataset_validation.e1_mask,
#                         'e2_mask': self.dataset_validation.e2_mask
#                     }
#
#                     for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch_val)):
#                         word_ids, pos_ids, synset_ids, triple_ids, head_mask, e1_mask, e2_mask, labels \
#                             = batch
#                         # positions_1, positions_2, word_ids, pos_ids, synset_ids, relation_ids, labels = batch
#                         acc, f1 = self._accuracy(sess, feed_dict={
#                             self.word_ids: word_ids,
#                             self.pos_ids: pos_ids,
#                             self.synset_ids: synset_ids,
#                             self.labels: labels,
#                             self.triple_ids: triple_ids,
#                             self.head_mask: head_mask,
#                             self.e1_mask: e1_mask,
#                             self.e2_mask: e2_mask
#                             # self.triple_ids: triple_ids,
#                         })
#                         total_f1.append(f1)
#
#                     val_f1 = np.mean(total_f1)
#                     Log.log("F1: {}".format(val_f1))
#                     print("Best F1: ", best_f1)
#                     print("F1 for epoch number {}: {}".format(e + 1, val_f1))
#                     if val_f1 > best_f1:
#                         saver.save(sess, self.model_name)
#                         Log.log('Save the model at epoch {}'.format(e + 1))
#                         best_f1 = val_f1
#                         n_epoch_no_improvement = 0
#                     else:
#                         n_epoch_no_improvement += 1
#                         Log.log("Number of epochs with no improvement: {}".format(n_epoch_no_improvement))
#                         if n_epoch_no_improvement >= patience:
#                             print("Best F1: {}".format(best_f1))
#                             break
#
#             if not early_stopping:
#                 saver.save(sess, self.model_name)
#
#     def _accuracy(self, sess, feed_dict):
#         feed_dict = feed_dict
#         logits = sess.run(self.logits, feed_dict=feed_dict)
#         accuracy = []
#         f1 = []
#         predict = []
#         exclude_label = []
#         for logit, label in zip(logits, feed_dict[self.labels]):
#             logit = np.argmax(logit)
#             exclude_label.append(label)
#             predict.append(logit)
#             accuracy += [logit == label]
#
#         f1.append(f1_score(predict, exclude_label, average='macro'))
#         return accuracy, np.mean(f1)
#
#     def load_data(self, train, validation):
#         timer = Timer()
#         timer.start("Loading data")
#
#         self.dataset_train = train
#         self.dataset_validation = validation
#
#         print("Number of training examples:", len(self.dataset_train.labels))
#         print("Number of validation examples:", len(self.dataset_validation.labels))
#         timer.stop()
#
#     def run_train(self, epochs, early_stopping=True, patience=10):
#         timer = Timer()
#         timer.start("Training model...")
#         self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
#         timer.stop()
#
#     def predict(self, test):
#         saver = tf.compat.v1.train.Saver()
#         with tf.compat.v1.Session() as sess:
#             Log.log("Testing model over test set")
#             saver.restore(sess, self.model_name)
#
#             y_pred = []
#             num_batch = len(test.labels) // self.batch_size + 1
#
#             data = {
#                 'words': test.words,
#                 'poses': test.poses,
#                 'synsets': test.synsets,
#                 'labels': test.labels,
#                 'triples': test.triples,
#                 'head_mask': test.head_mask,
#                 'e1_mask': test.e1_mask,
#                 'e2_mask': test.e2_mask
#             }
#
#             for idx, batch in enumerate(self._next_batch(data=data, num_batch=num_batch)):
#                 word_ids, pos_ids, synset_ids, triple_ids, head_mask, e1_mask, e2_mask, labels = batch
#
#                 feed_dict = {
#                     self.word_ids: word_ids,
#                     self.pos_ids: pos_ids,
#                     self.synset_ids: synset_ids,
#                     self.labels: labels,
#                     self.triple_ids: triple_ids,
#                     self.head_mask: head_mask,
#                     self.e1_mask: e1_mask,
#                     self.e2_mask: e2_mask
#                     # self.triple_ids: triple_ids,
#                 }
#                 logits = sess.run(self.logits, feed_dict=feed_dict)
#
#                 for logit in logits:
#                     decode_sequence = np.argmax(logit)
#                     y_pred.append(decode_sequence)
#
#         return y_pred
