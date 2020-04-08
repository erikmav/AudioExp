from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import keras.optimizers
import keras.utils
import numpy

# Saves the model to the file path (if specified). Returns a summary result.
def TrainAndValidateModel(
    mfccTensors,
    instrumentOneHotLabels,
    testMfccTensors,
    testInstrumentOneHotLabels,
    numConv1Filters,
    conv1KernelSize,
    numConv2Filters,
    conv2KernelSize,
    soundModelParams,
    numFullyConnectedPerceptronsLastLayer,
    numInstruments,
    batchSize = 16,
    epochs = 8,
    conv1Dropout = 0,
    conv2Dropout = 0,
    fullyConnectedDropout = 0,
    modelSavePath = None):
    
    print("TrainAndValidateModel:")
    print("  numConv1Filters:", numConv1Filters)
    print("  conv1KernelSize:", conv1KernelSize)
    print("  numConv2Filters:", numConv2Filters)
    print("  conv2KernelSize:", conv1KernelSize)
    print("  numFullyConnectedPerceptronsLastLayer:", numFullyConnectedPerceptronsLastLayer)
    print("  conv1Dropout:", conv1Dropout)
    print("  conv2Dropout:", conv2Dropout)
    print("  fullyConnectedDropout:", fullyConnectedDropout)

    # (numSamples, numRows, numColumns, numLayers)
    inputShape = numpy.shape(mfccTensors)
    inputShape = (inputShape[1], inputShape[2], inputShape[3])
    print("Input shape", inputShape)

    model = Sequential([
        # Layer 1: Inputs of MFCCs leading to the first convolutional layer.
        Conv2D(numConv1Filters, conv1KernelSize, kernel_initializer='TruncatedNormal', activation='relu', input_shape=inputShape, padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv1Dropout),

        # Layer 2: Convolution over results from conv layer 1. This provides an integration over a wider time period,
        # using the features extracted from the first layer.
        Conv2D(numConv2Filters, conv2KernelSize, kernel_initializer='TruncatedNormal', activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(conv2Dropout),

        # Reduce dimensionality before connecting to fully connected layers.
        Flatten(),

        # Layer 3: Fully connected layer with ReLU activation.
        Dense(numFullyConnectedPerceptronsLastLayer, activation='relu'),
        Dropout(fullyConnectedDropout),

        # Outputs: SoftMax activation to get probabilities by instrument.
        Dense(numInstruments, activation='softmax')
    ])

    print(model.summary())

    # Compile for categorization.
    model.compile(
        optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True),
        loss = 'categorical_crossentropy',
        metrics = [ 'accuracy' ])

    history = model.fit(mfccTensors, instrumentOneHotLabels, epochs=epochs, batch_size=batchSize)

    score = model.evaluate(testMfccTensors, testInstrumentOneHotLabels, batch_size=batchSize)
    print("Score:", model.metrics_names, score)

    result = {
        "training_" + model.metrics_names[0]: history.history[model.metrics_names[0]][epochs - 1],
        "training_" + model.metrics_names[1]: history.history[model.metrics_names[1]][epochs - 1],
        "testdata_" + model.metrics_names[0]: score[0],
        "testdata_" + model.metrics_names[1]: score[1],
        "batchSize": batchSize,
        "epochs": epochs,
        "maxpoolsize": (2, 2),
        "numConv1Filters": numConv1Filters,
        "conv1KernelSize": conv1KernelSize,
        "numConv2Filters": numConv2Filters,
        "conv2KernelSize": conv2KernelSize,
        "numFullyConnectedPerceptronsLastLayer": numFullyConnectedPerceptronsLastLayer,
        "conv1Dropout": conv1Dropout,
        "conv2Dropout": conv2Dropout,
        "fullyConnectedDropout": fullyConnectedDropout,
        "numInstruments": numInstruments,
        "numTrainingSamples": numpy.shape(mfccTensors)[0],
        "numTestSamples": numpy.shape(testMfccTensors)[0],
        "modelSavePath": modelSavePath
    }

    if modelSavePath is not None:
        model.save(modelSavePath)
        soundModelParams.save(modelSavePath + ".params.json")

    # Memory usage grows without bound if we don't delete as we go.
    del model

    return result
