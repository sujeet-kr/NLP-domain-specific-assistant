import tensorflow as tf
from tensorflow.contrib import layers

import timeline

import set_decoder
import project_helper as p_helper

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

def seq2seq(features, labels, mode, params):

    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    input_max_length = params['input_max_length']
    output_max_length = params['output_max_length']
    dropout = params['dropout']
    attention_mechanism_name = params['attention_mechanism_name']
    cell_type = params['cell_type']

    inp = features['input']
    batch_size = tf.shape(inp)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, 1)), 1)

    input_embed = layers.embed_sequence(
        inp, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')

    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    fw_cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    bw_cell = tf.contrib.rnn.GRUCell(num_units=num_units)

    if dropout > 0.0:
        print("  %s, dropout=%g " % (type(fw_cell).__name__, dropout))
        fw_cell = tf.contrib.rnn.DropoutWrapper(
            cell=fw_cell, input_keep_prob=(1.0 - dropout))
        bw_cell = tf.contrib.rnn.DropoutWrapper(
            cell=bw_cell, input_keep_prob=(1.0 - dropout))

    bd_encoder_outputs, bd_encoder_final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                        inputs=input_embed, dtype=tf.float32)

    encoder_outputs = tf.concat(bd_encoder_outputs, -1)
    encoder_final_state = tf.concat(bd_encoder_final_state, -1)

    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=END_TOKEN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Specific for Prediction
        pred_outputs = set_decoder.setting_decoder(pred_helper, 'decode', num_units, encoder_outputs, input_lengths,
                                    vocab_size, batch_size, output_max_length, attention_mechanism_name, cell_type,
                                    reuse=False)

        tf.identity(pred_outputs.sample_id[0], name='predictions')

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_outputs.sample_id)

    else:
        # Specific For Training
        output = features['output']
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)

        output_embed = layers.embed_sequence(
            train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)

        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)

        train_outputs = set_decoder.setting_decoder(train_helper, 'decode', num_units, encoder_outputs, input_lengths,
                        vocab_size, batch_size, output_max_length, attention_mechanism_name, cell_type, reuse=None)

        pred_outputs = set_decoder.setting_decoder(pred_helper, 'decode', num_units, encoder_outputs, input_lengths,
                        vocab_size, batch_size, output_max_length, attention_mechanism_name, cell_type, reuse=True)

        tf.identity(train_outputs.sample_id[0], name='train_pred')
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))

        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, output, weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.get('optimizer', 'Adam'),
            learning_rate=params.get('learning_rate', 0.001),
            summaries=['loss', 'learning_rate'])

        tf.identity(pred_outputs.sample_id[0], name='predictions')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_outputs.sample_id,
            loss=loss,
            train_op=train_op
        )



def train_seq2seq(
        input_filename, output_filename, vocab_filename,
        model_dir):
    vocab = p_helper.load_vocab(vocab_filename)
    params = {
        'vocab_size': len(vocab),
        'batch_size': 128,
        'input_max_length': 20,
        'output_max_length': 20,
        'embed_dim': 100,
        'num_units': 256,
        'dropout': 0.2,
        'attention_mechanism_name': 'scaled_luong',
        'cell_type': 'GRU'
    }
    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, params=params)

    input_fn, feed_fn = p_helper.make_input_fn(
        params['batch_size'],
        input_filename,
        output_filename,
        vocab, params['input_max_length'], params['output_max_length'])


    # Make hooks to print examples of inputs/predictions.
    print_inputs = tf.train.LoggingTensorHook(
        ['input_0', 'output_0'], every_n_iter=100,
        formatter=p_helper.get_formatter(['input_0', 'output_0'], vocab))
    print_predictions = tf.train.LoggingTensorHook(
        ['predictions', 'train_pred'], every_n_iter=100,
        formatter=p_helper.get_formatter(['predictions', 'train_pred'], vocab))

    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)

    est.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_predictions,
               timeline_hook],
        steps=10000)



def predict_seq2seq(input_filename, vocab_file, model_dir, input_mode):
    vocab = p_helper.load_vocab(vocab_file)

    params = {
        'vocab_size': len(vocab),
        'batch_size': 3,
        'embed_dim': 100,
        'num_units': 256,
        'input_max_length': 20,
        'output_max_length': 20,
        'dropout': 0.0,
        'attention_mechanism_name': 'scaled_luong',
        'cell_type': 'GRU'
    }

    model = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, params=params)

    inputs_with_tokens = p_helper.predict_input_fn(input_filename, vocab, input_mode)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=inputs_with_tokens,
                                                       shuffle=False,
                                                       num_epochs=1)

    predictions_obj = model.predict(input_fn=pred_input_fn)
    final_answer = p_helper.get_out_put_from_tokens(predictions_obj, vocab)

    if input_mode.upper() == 'INPUT_FILE':
        with open(input_filename) as finput:
            for each_answer in final_answer:
                question = finput.readline()
                print('Question: ', question.replace('\n','').replace('<EOS>',''))
                print('Answer', str(each_answer).replace('<EOS>','').replace('<GO>',''))

    elif input_mode.upper() == 'API':
        return (str(final_answer(0)).replace('<EOS>','').replace('<GO>', ''))

    else:
        for each_answer in final_answer:
            print('Question: ', str(p_helper.get_out_put_from_tokens(
                            inputs_with_tokens['input'], vocab)[0]).replace('<EOS>',''))
            print('Answer', str(each_answer).replace('<EOS>',''))
