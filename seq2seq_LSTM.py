import os

from scipy._lib.six import xrange

os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'
#--
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys 
import copy
import random

from Code.fileprocessor import *
from Code.preprocessor import *
from Code.calculateError import *

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

#-----------------------------------------------------
def main(set_path,x_l,y_l):
# obtaining the Data ------------------------

    x_length = int(x_l) # the input sequence length
    y_length = int(y_l) # the output sequence length
    percentage = 0.8 # the percentage of data used for training
    filename = './Dataset/'+set_path+".txt" #Set this to the dataset on which the model is to be run
    name_flag = "result" # the name flag for the test case
    save_path_name = os.getcwd() # the pwd to current directory
    save_object_name = name_flag # the state name to be saved

    X_train_data, Y_train_data, X_test_data, Y_test_data = getData(filename,x_length,y_length,percentage)

    X_train = np.array(X_train_data)
    Y_train = np.array(Y_train_data)
    X_test = np.array(X_test_data)
    Y_test = np.array(Y_test_data)

    #----- create a new random sample from training set ---

    X_train_random_data = []
    Y_train_random_data = []
    sample_percentage = 0.1 # 10% of the train sample is selected
    sample_size = int(round(len(X_train_data)*sample_percentage))
    indices = random.sample(xrange(len(X_train_data)),sample_size)

    for i in range(len(indices)):
        X_train_random_data.append(X_train_data[int(indices[i])])
        Y_train_random_data.append(Y_train_data[int(indices[i])])

    X_train_random = np.array(X_train_random_data)
    Y_train_random = np.array(Y_train_random_data)

    name = "Seq2Seq_unguided_"+name_flag+"_LSTM_Y_test_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
    writetofile(name,Y_test_data)

    #--------------------------------------------

    learning_rate = 0.01 # learning rate parameter
    lambda_l2_reg = 0.003 # l2 regularization parameter

    hidden_size = 200 # LSTM hidden node size
    input_dim = 1 # the numeber of input signals
    output_dim = 1 # the number of output signals

    num_stacked_layers = 2 # 2 stacked layers
    gradient_clipping = 2.5 # gradient clipping parameter

# ---------------------------------------------

    # when feed_previous = True, the decoder uses the previous output as an input
    def graph(feed_previous=False):
        tf.reset_default_graph()  # resets the previous graph

        global_step = tf.Variable(initial_value=0, name="global_step", trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        weights = {
            'out': tf.get_variable('Weights_out', shape=[hidden_size, output_dim], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer()),
        }

        biases = {
            'out': tf.get_variable('Biases_out', shape=[output_dim], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.)),
        }

        with tf.variable_scope('Seq2seq'):
            # Encoder : inputs
            enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                for t in range(x_length)
            ]

            # Decoder : target outputs
            target_seq = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
                for t in range(y_length)
            ]

            # -- new method
            # instead of giing an END symbol, instead input the last value that was in the sequence given
            # as the first input the the decoder

            dec_inp = [enc_inp[-1]] + target_seq[:-1]

            # -- building the LSTM cell
            with tf.variable_scope('LSTMCell'):
                cells = []
                for i in range(num_stacked_layers):
                    with tf.variable_scope('RNN_{}'.format(i)):
                        cells.append(tf.contrib.rnn.LSTMCell(hidden_size))
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            def _rnn_decoder(decoder_inputs,
                             initial_state,
                             cell,
                             loop_function=None,
                             scope=None):

                with variable_scope.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        if loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output
                return outputs, state

            def _basic_rnn_seq2seq(encoder_inputs,
                                   decoder_inputs,
                                   cell,
                                   feed_previous,
                                   dtype=dtypes.float32,
                                   scope=None):

                with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                    enc_cell = copy.deepcopy(cell)
                    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                    if feed_previous:
                        return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                    else:
                        return _rnn_decoder(decoder_inputs, enc_state, cell)

            def _loop_function(prev, _):
                return tf.matmul(prev, weights['out']) + biases['out']

            dec_outputs, dec_memory = _basic_rnn_seq2seq(enc_inp, dec_inp, cell, feed_previous=feed_previous)
            reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

        # Training loss and optimizer
        with tf.variable_scope('Loss'):
            # L2 loss
            output_loss = tf.reduce_mean(tf.squared_difference(reshaped_outputs, target_seq))
            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            loss = output_loss + lambda_l2_reg * reg_loss

        with tf.variable_scope('Optimizer'):
            optimizer = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=learning_rate, global_step=global_step,
                                                        optimizer='Adam', clip_gradients=gradient_clipping)

        saver = tf.train.Saver

        return dict(enc_inp=enc_inp, target_seq=target_seq, train_op=optimizer, loss=loss, saver=saver,
                    reshaped_outputs=reshaped_outputs)

    # un-guided training method
    ep = 0;
    loss_t = 300
    avg_rmse_lim = 0.1
    LOSS_LIMIT = avg_rmse_lim * avg_rmse_lim
    CONTINUE_FLAG = True
    EPOCH_LIMIT = 20

    rnn_model =graph(feed_previous=True) #un-guided training model

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    print('total_params:{}'.format(sum([np.prod(v.get_shape().as_list())for v in tf.trainable_variables()])))
    Y_found = []
    train_loss = []
    train_RMSE = []

    past_loss_values = []
    epoch_range = 5

    with tf.Session() as sess:
        print ("--- tensorflow session started ---")
        init.run()
        # -- training
        while CONTINUE_FLAG:
            #-----------------------------------
            feed_dict = {rnn_model['enc_inp'][t]:X_train[:,t].reshape(-1,input_dim) for t in range(x_length)}
            feed_dict.update({rnn_model['target_seq'][t]:Y_train[:,t].reshape(-1,output_dim) for t in range(y_length)})
            train_t,loss_t,out_t = sess.run([rnn_model['train_op'],rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict)
            train_loss.append(loss_t)
            if ep % 10 == 0:
                temp_output = np.reshape(out_t,(y_length,-1))
                temp_output = temp_output.transpose()
                temp_y_found = temp_output.tolist()
                temp_err = RMSE(Y_train_data,temp_y_found)
                train_RMSE.append(temp_err)
                print (ep," loss :",loss_t ," output size :",np.array(out_t).shape)
                #-------------------- STATE LOGGER--------------------------------
                # log state of identified values every 2000 epochs
                if ep % 100 == 0:
                    print ("-- state logged @ epoch :",ep)
                    name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_RMSE_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
                    # writetofile(name,train_RMSE) # saving the train RMSE values for every 200th epoch

                    name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_LOSS_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
                    # writeErrResult(name,train_loss) # append the train loss

                    temp_saver = rnn_model['saver']()
                    save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))
                #-----------------------------------------------------------------
            #-- condition to stop training
            #-- condition to keep track of past losses
            if ep < epoch_range:
                past_loss_values.append(loss_t)
            else:
                past_loss_values.pop(0)
                past_loss_values.append(loss_t)
            # increase the epoch count
            ep += 1
            #-- find if the entire range of previous losses are below a threshold
            count = 0
            for val in past_loss_values:
                if val < LOSS_LIMIT:
                    count += 1
            #-- stopping condition for training
            if count >= epoch_range or ep >= EPOCH_LIMIT:
                CONTINUE_FLAG = False
        print ("-- training stopped @ epoch :",ep )
        print ("--- randomized training started ---")
        CONTINUE_FLAG = True # reset the continue flag
        while CONTINUE_FLAG:
            #-----------------------------------
            feed_dict = {rnn_model['enc_inp'][t]:X_train_random[:,t].reshape(-1,input_dim) for t in range(x_length)}
            feed_dict.update({rnn_model['target_seq'][t]:Y_train_random[:,t].reshape(-1,output_dim) for t in range(y_length)})
            train_t,loss_t,out_t = sess.run([rnn_model['train_op'],rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict)
            train_loss.append(loss_t)
            if ep % 10 == 0:
                temp_output = np.reshape(out_t,(y_length,-1))
                temp_output = temp_output.transpose()
                temp_y_found = temp_output.tolist()
                temp_err = RMSE(Y_train_random,temp_y_found)
                train_RMSE.append(temp_err)
                print(ep," loss :",loss_t ," output size :",np.array(out_t).shape)
                #-------------------- STATE LOGGER--------------------------------
                # log state of identified values every 2000 epochs
                if ep % 100 == 0:
                    print ("-- state logged @ epoch :",ep)
                    name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_RMSE_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
                    # writetofile(name,train_RMSE) # saving the train RMSE values for every 200th epoch

                    name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_LOSS_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
                    # writeErrResult(name,train_loss) # append the train loss

                    temp_saver = rnn_model['saver']()
                    save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))
                #-----------------------------------------------------------------
            #-- condition to stop training
            #-- condition to keep track of past losses
            if ep < epoch_range:
                past_loss_values.append(loss_t)
            else:
                past_loss_values.pop(0)
                past_loss_values.append(loss_t)
            # increase the epoch count
            ep += 1
            #-- find if the entire range of previous losses are below a threshold
            count = 0
            for val in past_loss_values:
                if val < LOSS_LIMIT:
                    count += 1
            #-- stopping condition for training
            if count >= epoch_range or ep >= EPOCH_LIMIT:
                CONTINUE_FLAG = False
        print ("-- randomized training stopped @ epoch :",ep)

        name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_RMSE_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
        writetofile(name,train_RMSE) # saving the train RMSE values for every 200th epoch
        name =  "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_LOSS_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
        #writeErrResult(name,train_loss) # append the train loss

        print ("--- training complete ---")
        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))
        print ("--- session saved ---")

        loss_t,out_t = sess.run([rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict)
        temp_output = np.reshape(out_t,(y_length,-1))
        temp_output = temp_output.transpose()
        temp_y_found = temp_output.tolist()
        temp_err = RMSE(Y_train_random,temp_y_found)
        name = "Seq2seq_unguided_"+name_flag+"_LSTM_TRAIN_FOUND_RMSE_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
        #writeErrResult(name,temp_err)

        print ("--- testing started ---")
        feed_dict2 = {rnn_model['enc_inp'][t]:X_test[:,t].reshape(-1,input_dim) for t in range(x_length)}

        Y_temp = np.zeros((len(X_test),y_length), dtype=np.float)
        feed_dict2.update({rnn_model['target_seq'][t]:Y_temp[:,t].reshape(-1,output_dim) for t in range(y_length)})
        #--
        #print np.array(rnn_model['reshaped_outputs']).shape
        out_t = sess.run([rnn_model['reshaped_outputs']],feed_dict2)
        print ("prediction size: ", np.array(out_t).shape)
        matrix = np.reshape(out_t,(y_length,-1))
        print ("reshaped output: ", matrix.shape)
        matrix = matrix.transpose()
        print ("transposed matrix: ",matrix.shape)
        Y_found = matrix.tolist()
        # -- testing

    #------- saving the outputs of Y from testing
    name = "Seq2seq_unguided_"+name_flag+"_LSTM_Y_found_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
    writetofile(name,Y_found)

    err = RMSE(Y_test_data,Y_found)

    name = "Seq2seq_unguided_"+name_flag+"_LSTM_RMSE_x_"+str(x_length)+"_y_"+str(y_length)+"data.txt"
    writeErrResult(name,err)

    print ("----- run complete-------")

