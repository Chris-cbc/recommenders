# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.dkn import DKN
import numpy as np
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric

r"""
This new model adapts DKN's structure for item-to-item recommendations.
The tutorial can be found at: https://github.com/microsoft/recommenders/blob/kdd2020_tutorial/scenarios/academic/KDD2020-tutorial/step4_run_dkn_item2item.ipynb
 """


class DKNItem2Item(DKN):
    def _compute_data_loss(self):
        logits = self.pred
        data_loss = -1 * tf.reduce_sum(input_tensor=tf.math.log(logits[:, 0] + 1e-10))
        return data_loss

    def _build_dkn(self):
        """The main function to create DKN's logic.

        Returns:
            obj: Prediction of item2item relation scores made by the DKN model, in the shape of (batch_size, num_negative + 1).
        """
        news_field_embed_final_batch = self._build_doc_embedding(
            self.iterator.candidate_news_index_batch,
            self.iterator.candidate_news_entity_index_batch,
        )

        self.news_field_embed_final_batch = tf.math.l2_normalize(
            news_field_embed_final_batch, axis=-1, epsilon=1e-12
        )

        item_embs_train = tf.reshape(
            self.news_field_embed_final_batch,
            [
                -1,
                self.iterator.neg_num + 2,
                self.news_field_embed_final_batch.shape[-1],
            ],
        )  # (B, group, D)

        item_embs_source = item_embs_train[:, 0, :]  # get the source item
        item_embs_source = tf.expand_dims(item_embs_source, 1)

        item_embs_target = item_embs_train[:, 1:, :]

        item_relation = tf.math.multiply(item_embs_target, item_embs_source)
        item_relation = tf.reduce_sum(input_tensor=item_relation, axis=-1)  # (B, neg_num + 1)

        self.pred_logits = item_relation

        return self.pred_logits

    def _get_pred(self, logit, task):
        return tf.nn.softmax(logit, axis=-1)

    def _build_doc_embedding(self, candidate_word_batch, candidate_entity_batch):
        """
        To make the document embedding be dense, we add one tanh layer on top of the kims_cnn module.
        """
        with tf.compat.v1.variable_scope("kcnn", initializer=self.initializer):
            news_field_embed = self._kims_cnn(
                candidate_word_batch, candidate_entity_batch, self.hparams
            )
            W = tf.compat.v1.get_variable(
                name="W_doc_trans",
                shape=(news_field_embed.shape[-1], self.num_filters_total),
                dtype=tf.float32,
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution=(
                    "uniform" if False else "truncated_normal")),
            )
            if W not in self.layer_params:
                self.layer_params.append(W)
            news_field_embed = tf.tanh(tf.matmul(news_field_embed, W))
        return news_field_embed

    def eval(self, sess, feed_dict):
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        preds = sess.run(self.pred, feed_dict=feed_dict)
        labels = np.zeros_like(preds, dtype=np.int32)
        labels[:, 0] = 1
        return (preds, labels)

    def run_eval(self, filename):
        load_sess = self.sess
        group_preds = []
        group_labels = []

        for (
            batch_data_input,
            newsid_list,
            data_size,
        ) in self.iterator.load_data_from_file(filename):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                group_preds.extend(step_pred)
                group_labels.extend(step_labels)

        res = cal_metric(group_labels, group_preds, self.hparams.pairwise_metrics)
        return res
