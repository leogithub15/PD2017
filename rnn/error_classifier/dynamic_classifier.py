import tensorflow as tf

class DynamicClassifier:

    def __init__(self, vocab_size, input_embedding_size, output_classes, num_units, num_layers):

        #self.inputs = tf.placeholder(tf.float32, (None, None, vocab_size))
        self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        input_embedded = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.sequence_length = tf.placeholder(shape=(None,), dtype=tf.int32)

        self.outputs = tf.placeholder(tf.float32, (None, output_classes))
        self.keep_prob = tf.placeholder(tf.float32)


        # Create LSTM cells with droput
        stacked_rnn_fw = []
        stacked_rnn_bw = []
        for i in range(num_layers):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)

            # Add a dropout layer
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.keep_prob)

            stacked_rnn_fw.append(cell_fw)
            stacked_rnn_bw.append(cell_bw)

        # Create the stacked cells
        multylayer_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
        multylayer_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        ((fw_outputs,
          bw_outputs),
         (fw_final_state,
          bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=multylayer_cell_fw,
                                                             cell_bw=multylayer_cell_bw,
                                                             inputs=input_embedded,
                                                             sequence_length=self.sequence_length,
                                                             dtype=tf.float32)

        # Select last output.
        #rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
        #last_outputs = tf.gather(rnn_outputs, 20, axis=1)

        last_outputs = tf.concat((fw_final_state[num_layers-1].h,
                                  bw_final_state[num_layers-1].h), 1)

        # Create a fully connected layer
        logits = tf.contrib.layers.fully_connected(last_outputs, output_classes, activation_fn=None)

        # Prediction function
        self.prediction = tf.nn.softmax(logits)

        # Error function
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.outputs))

        # Train function
        self.train = tf.train.AdamOptimizer().minimize(self.error)

        # Model evaluation
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.outputs, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))