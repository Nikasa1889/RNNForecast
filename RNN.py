import os.path
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from loadData import loadData, convertToBatches
import logging
__version__ = "1.0.0"
class RNN(object):
    """Recursive Neural Network 2 layers without CNN as feature extractor"""
    
    def __init__(self, maxGradient, timeSteps, nHorizons, inputSize, nHiddenUnits, nLayers):
        self.maxGradient = maxGradient
        self.nLayers = nLayers
        self.timeSteps = timeSteps
        self.nHorizons = nHorizons
        self.inputSize = inputSize
        self.nHiddenUnits = nHiddenUnits

        with tf.name_scope("Parameters"):
            self.learningRate = tf.placeholder(tf.float32, name="learningRate")
            self.keepProbability = tf.placeholder(tf.float32, name="keepProbability")
            
        with tf.name_scope("Input"):
            self.input = tf.placeholder(tf.float32, shape=(None, timeSteps, inputSize), name="input")
            self.targets = tf.placeholder(tf.float32, shape=(None, timeSteps, nHorizons), name="targets")
            self.init = tf.placeholder(tf.float32, shape=(), name="init")
            self.batchSize = self.input.get_shape()[0]
        #Declare the CNN structure here!
        #with tf.name_scope("Embedding"):
        #    self.embedding = tf.Variable(tf.random_uniform((inputSize, hidden_units), -self.init, self.init),
        #                                 dtype=tf.float32,
        #                                 name="embedding")
        #    self.w = tf.get_variable("w", (inputSize, hidden_units))
        #    self.b = tf.get_variable("b", inputSize)
            
        #    self.embedded_input = tf.matmul(self.input, self.w) + self.b

        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.LSTMCell(nHiddenUnits, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keepProbability)
            self.rnn_layers = tf.nn.rnn_cell.MultiRNNCell([cell] * nLayers, state_is_tuple=True)
            state_placeholder = tf.placeholder(tf.float32, [nLayers, 2, None, nHiddenUnits])
            #Unpack the state_placeholder into tuple to use with tensorflow native RNN API
            l = tf.unpack(state_placeholder, axis=0)
            self.state = tuple(
                                [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1]) 
                                for idx in range(nLayers)]
                              )
            
            self.outputs, self.nextState = tf.nn.dynamic_rnn(self.rnn_layers, self.input, time_major=False,
                                                              initial_state=self.state)

        with tf.name_scope("Cost"):
            # Concatenate all the batches into a single row.
            self.flattenedOutputs = tf.reshape(self.outputs, (-1, nHiddenUnits),
                                                name="flattenedOutputs")
            # Project the outputs onto the vocabulary.
            self.w = tf.get_variable("w", (nHiddenUnits, nHorizons))
            self.b = tf.get_variable("b", nHorizons)
            self.predicted = tf.matmul(self.flattenedOutputs, self.w) + self.b
            self.flattenedTargets = tf.reshape(self.targets, (-1, nHorizons), name = "flattenedTargets")
            # Compare predictions to labels.
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.flattenedTargets, self.predicted))))
            self.cost = tf.reduce_mean(self.loss, name="cost")

        with tf.name_scope("Train"):
            tf.scalar_summary('RMSE', self.cost)
            self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                       maxGradient, name="clipGradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate)
            self.trainStep = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                        name="trainStep",
                                                        global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()
        self.summary = tf.merge_all_summaries()

    def train(self, session, init, ts, parameters, exitCriteria, validation, loggingInterval, directories, logger):
        epoch = 1
        iteration = 0
        state = None
        trainingSet = convertToBatches(ts, self.timeSteps, parameters.batchSize, self.nHorizons)
        self.resetState = self.rnn_layers.zero_state(parameters.batchSize, dtype=tf.float32)
        if not logger:
            self.logger = logging.getLogger('dummy')
        else:
            self.logger = logger
        summaryWriter = self.summaryWriter(directories.summary, session)
        session.run(self.initialize, feed_dict={self.init: init})
        try:
            # Enumerate over the training set until exit criteria are met.
            tsFit = []
            tsTarget = []
            lastState = None
            while True:
                if (exitCriteria.maxEpochs is not None) and (epoch > exitCriteria.maxEpochs):
                    lastState = state
                    raise StopTrainingException()
                epochCost = epochIteration = 0
                #Reset state after every epoch
                state = session.run(self.resetState)
                # Enumerate over a single epoch of the training set.
                for xs, ys in trainingSet:
                    _, summary, cost, state, iteration, predicted = session.run(
                        [self.trainStep, self.summary, self.cost, self.nextState, self.iteration, self.predicted],
                        feed_dict={
                            self.input: xs,
                            self.targets: ys,
                            self.state: state,
                            self.learningRate: parameters.learningRate,
                            self.keepProbability: parameters.keepProbability
                        })
                    if (epoch == exitCriteria.maxEpochs):
                        tsFit.append(predicted)
                        tsTarget.append(ys)
                        
                    epochCost += cost
                    epochIteration += self.timeSteps
                    if self._interval(iteration, loggingInterval):
                        self.logger.info("Epoch %d, Iteration %d: training loss %0.4f" %
                                (epoch, iteration, cost))
                    summaryWriter.add_summary(summary, iteration)
                    

                self.logger.info("---Epoch %d, Iteration %d: epoch loss %0.4f" % (epoch, iteration, epochCost))

                epoch += 1
                if (exitCriteria.maxIterations is not None) and (iteration > exitCriteria.maxIterations):
                    raise StopTrainingException()
        except (StopTrainingException, KeyboardInterrupt):
            pass
        self.logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
        summaryWriter.close()
        if directories.model is not None:
            modelFileName = self._modelFile(directories.model)
            tf.train.Saver().save(session, modelFileName)
            self._writeModelParameters(directories.model)
            self.logger.info("Saved model in %s " % directories.model)
            
        tsFit = np.reshape(np.asarray(tsFit), (-1, self.nHorizons))
        tsTarget = np.reshape(np.asarray(tsTarget), (-1, self.nHorizons))
        return (tsTarget, tsFit, lastState)

    def _writeModelParameters(self, modelDirectory):
        parameters = {
            "maxGradient": self.maxGradient,
            "timeSteps": self.timeSteps,
            "inputSize": self.inputSize,
            "nHiddenUnits": self.nHiddenUnits,
            "nLayers": self.nLayers,
            "nHorizons": self.nHorizons
        }
        with open(self._parametersFile(modelDirectory), "w") as f:
            json.dump(parameters, f, indent=4)

    def predict(self, session, startState, tsTest, batchSize):
        state = None
        testSet = convertToBatches(tsTest, self.timeSteps, batchSize, self.nHorizons, cutHead=False)
        tsPredicted = []
        tsTarget = []
        epoch_cost = 0
        state = startState
        for xs, ys in testSet:
            cost, state, predicted = session.run(
                [self.cost, self.nextState, self.predicted],
                feed_dict={
                    self.input: xs,
                    self.targets: ys,
                    self.state: state,
                    self.keepProbability: 1
                })
            epoch_cost += cost
            tsPredicted.append(predicted)
            tsTarget.append(ys)
        
        tsPredicted = np.reshape(np.asarray(tsPredicted), (-1, self.nHorizons))
        tsTarget = np.reshape(np.asarray(tsTarget), (-1, self.nHorizons))
        return (tsTarget, tsPredicted, epoch_cost)

    @staticmethod
    def _interval(iteration, interval):
        return interval is not None and iteration > 1 and iteration % interval == 0

    @staticmethod
    def summaryWriter(summaryDirectory, session):
        class NullSummaryWriter(object):
            def addSummary(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        if summaryDirectory is not None:
            return tf.train.SummaryWriter(summaryDirectory, session.graph)
        else:
            return NullSummaryWriter()

    @classmethod
    def restore(cls, session, modelDirectory):
        """
        Restore a previously trained model
        :param session: session into which to restore the model
        :type session: TensorFlow Session
        :param model_directory: directory to which the model was saved
        :type model_directory: str
        :return: trained model
        :rtype: RNN
        """
        with open(cls._parametersFile(modelDirectory)) as f:
            parameters = json.load(f)
        model = cls(parameters["maxGradient"], parameters["timeSteps"], parameters["nHorizons"], 
                    parameters["inputSize"],parameters["nHiddenUnits"], parameters["nLayers"])
        tf.train.Saver().restore(session, cls._modelFile(modelDirectory))
        return model

    @staticmethod
    def _parametersFile(modelDirectory):
        return os.path.join(modelDirectory, "parameters.json")

    @staticmethod
    def _modelFile(modelDirectory):
        return os.path.join(modelDirectory, "model")
    @property
    def batch_size(self):
        return self.input.get_shape()[0].value

    @property
    def time_steps(self):
        return self.input.get_shape()[1].value

    @property
    def input_size(self):
        return self.input.get_shape()[2].value

    @property
    def hidden_units(self):
        return self.w.get_shape()[0].value

class StopTrainingException(Exception):
    pass


# Objects used to group training parameters
class ExitCriteria(object):
    def __init__(self, maxIterations, maxEpochs):
        self.maxIterations = maxIterations
        self.maxEpochs = maxEpochs


class Parameters(object):
    def __init__(self, learningRate, keepProbability, batchSize):
        self.learningRate = learningRate
        self.keepProbability = keepProbability
        self.batchSize = batchSize


class Validation(object):
    def __init__(self, interval, validation_set):
        self.interval = interval
        self.validation_set = validation_set


class Directories(object):
    def __init__(self, model, summary):
        self.model = model
        self.summary = summary