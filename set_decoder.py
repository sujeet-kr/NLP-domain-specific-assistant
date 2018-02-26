import tensorflow as tf

def setting_decoder(helper, scope, num_units, encoder_outputs, input_lengths,
            vocab_size, batch_size, output_max_length, attention_mechanism_name, cell_type, reuse=None):

    # attention_mechanism_name == Luong/Bahdanau/Scaled_Luong/Normalized_Bahdanau
    # cell_type == LSTM/GRU


    with tf.variable_scope(scope, reuse=reuse):
        #Selecting the Attention Mechanism
        if attention_mechanism_name.upper() == 'LUONG':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
        elif attention_mechanism_name.upper() == 'BAHDANAU':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths)
        elif attention_mechanism_name.upper() == 'SCALED_LUONG':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths, scale=True)
        elif attention_mechanism_name.upper() == 'NORMALIZED_BAHDANAU':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                memory_sequence_length=input_lengths, normalize=True)
        else:
            raise ValueError("Error in type of Attention Mechanism provided %s " % attention_mechanism_name)

        #Selecting the Cell Type to use
        if cell_type.upper() == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        elif cell_type.upper() == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=num_units)
        else:
            raise ValueError("Error in type of Cell Type provided %s " % cell_type)

        #Wrapping attention to the cell
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=num_units / 2)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, vocab_size, reuse=reuse
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size))
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=output_max_length
        )
        return outputs[0]



        # decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        #     encoder_state, multiplier=beam_width)
        #
        #
        #
        #
        # decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        #         cell=out_cell,
        #         embedding=embedding_decoder,
        #         start_tokens=start_tokens,
        #         end_token=end_token,
        #         initial_state=decoder_initial_state,
        #         beam_width=beam_width,
        #         output_layer=projection_layer,
        #         length_penalty_weight=0.0)



