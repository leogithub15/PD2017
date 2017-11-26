import tensorflow as tf
import numpy as np
import time as t
from dynamic_classifier import DynamicClassifier
from data_utility import load_data, one_hot_encode, emb_encode

def main():
    vocabulary_input = "abcdefghijklmnopqrtsuvwxyz"
    vocabulary_target = "0123"

    training_steps = 1000000
    display_step = 1000
    batch_size = 32
    vocab_size = 27
    input_embedded_size = 27
    output_classes = 4
    num_units = 1024
    num_layers = 2
    DROPOUT_KEEP_PROB_TRAIN = 1.0
    DROPOUT_KEEP_PROB_TEST = 1.0


    #file_name = "dataset_diff_lengths_test.txt"
    file_name = "datasetC.txt"
    #file_name = "dset.txt"
    ratio = np.array([0.8, 0.9])
    columns = np.array([1, 2])
    X_train, X_train_lengths, X_val, X_val_lengths, X_test, X_test_lengths, y_train, y_val, y_test,train_batch_sizes,test_batch_sizes = load_data(
        file_name, ratio, columns, batch_size)

    X_train_emb = emb_encode(X_train, vocabulary_input, train_batch_sizes,batch_size)
    X_test_emb = emb_encode(X_test, vocabulary_input, test_batch_sizes, batch_size )

    y_train_one_hot = one_hot_encode(y_train, vocabulary_target)
    y_train_one_hot = np.squeeze(y_train_one_hot, axis=1)
    y_test_one_hot = one_hot_encode(y_test, vocabulary_target)
    y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)


    model = DynamicClassifier(vocab_size, input_embedded_size, output_classes, num_units, num_layers)
    iteration = 0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        try:
            start = t.time()
            for step in range(1, training_steps + 1):
                batch_x = X_train_emb[batch_size * iteration:batch_size * (iteration + 1), :]
                batch_len_x = X_train_lengths[batch_size * iteration:batch_size * (iteration + 1)]
                batch_y = y_train_one_hot[batch_size * iteration:batch_size * (iteration + 1), :]
                iteration += 1
                if iteration >= (X_train_emb.shape[0] / batch_size) - 1:
                    iteration = 0

                err, _, acc, = sess.run([model.error, model.train, model.accuracy],
                                      feed_dict={model.inputs: batch_x,
                                                 model.outputs: batch_y,
                                                 model.keep_prob: DROPOUT_KEEP_PROB_TRAIN,
                                                 model.sequence_length: batch_len_x
                                                 })

                if step % display_step == 0 or step == 1:

                    # Calculate batch loss and accuracy
                    err, acc, = sess.run([model.error, model.accuracy],
                                            feed_dict={model.inputs: batch_x,
                                                       model.outputs: batch_y,
                                                       model.keep_prob: DROPOUT_KEEP_PROB_TRAIN,
                                                       model.sequence_length: batch_len_x
                                                       })
                    end = t.time()
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(err) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                    print("Epoch Time: ", end - start)
                    start = t.time()
                #print('step')

        except KeyboardInterrupt:
            print('training interrupted')

        batch_x = X_test_emb[0:5000, :]
        batch_len_x = X_test_lengths[0:5000]
        batch_y = y_test_one_hot[0:5000, :]
        acc = sess.run(model.accuracy, feed_dict={model.inputs: batch_x,
                                                  model.outputs: batch_y,
                                                  model.keep_prob: DROPOUT_KEEP_PROB_TEST,
                                                  model.sequence_length: batch_len_x
                                                  })
        print("Testing Accuracy:" + str(acc))


if __name__ == '__main__':
    main()