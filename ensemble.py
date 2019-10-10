import numpy as np
import imp
from keras.utils import to_categorical

class VotingModel(object):

    def __init__(self, model_list, voting='hard',
                 weights=None, nb_classes=None):

        self.model_list = model_list
        self.voting = voting
        self.weights = weights
        self.nb_classes = nb_classes

        if voting not in ['hard', 'soft']:
            raise 'Voting has to be either hard or soft'

        if weights is not None:
            if len(weights) != len(model_list):
                raise ('Number of models {0} and length of weight vector {1} has to match.'
                       .format(len(weights), len(model_list)))

    def predict(self, X, batch_size=128, verbose=0):
        predictions = list(map(lambda model: model.predict(X, batch_size, verbose)[0], self.model_list))
        nb_preds = len(X)
        print(np.array(predictions[0]).shape)
        if self.voting == 'hard':
            for i, pred in enumerate(predictions):
                
                pred = list(map(
                    lambda probas: np.argmax(probas, axis=-1), pred
                ))
                predictions[i] = np.asarray(pred).reshape(nb_preds, 1)
            argmax_list = list(np.concatenate(predictions, axis=1))
            votes = np.asarray(list(
                map(lambda arr: max(set(arr)), argmax_list)
            ))
        if self.voting == 'soft':
            for i, pred in enumerate(predictions):
                pred = list(map(lambda probas: probas * self.weights[i], pred))
                predictions[i] = np.asarray(pred).reshape(nb_preds, self.nb_classes, 1)
            weighted_preds = np.concatenate(predictions, axis=2)
            weighted_avg = np.mean(weighted_preds, axis=2)
            votes = np.argmax(weighted_avg, axis=1)

        return votes

def load_cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    # return (x_train[:1000], y_train[:1000]), (x_test[:100], y_test[:100])
    return (x_train, y_train), (x_test, y_test)



def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models_check]
    y = Average()(outputs)
    model = Model([model.input for model in models_check], y, name='ensemble')
    return model

(x_train, y_train), (x_test, y_test) = load_cifar10()
# x_test64 = np.load("../../x_test.npy")
t = np.argmax(y_test, 1)


def resize(data_set):
    X_temp = []
    import scipy
    for i in range(data_set.shape[0]):
        resized = scipy.misc.imresize(data_set[i], (64, 64))
        X_temp.append(resized)
    X_temp = np.array(X_temp, dtype=np.float32) / 255.
    return X_temp

x_test64 = resize(x_test)
# x_test64 = np.load("x_test.npy")

loop_size=7

for i in range(loop_size):

     m1 = imp.load_source('module.name', 'model/CIFAR10/'+ str(i+1)+'/deepcaps.py')
     _, eval_model1 = m1.DeepCapsNet(input_shape=x_test64.shape[1:], n_class=10, routings=3)
     eval_model1.load_weights('model/CIFAR10/'+str(i+1)+'/best_weights.h5')
     a1, b1 = eval_model1.predict(x_test64)
     np.save("model/CIFAR10/deepcaps_"+str(i+1)+".npy", a1)

d = [None] * (loop_size+1)
for i in range(1, loop_size+1):
     d[i] = np.load("model/CIFAR10/deepcaps_"+str(i)+".npy")
     p = np.sum(np.argmax(d[i], 1) == t) / y_test.shape[0]
     print(f'p{i} Test acc:', p)

# The following is a workaround for d8 and d9 in the next average
d.append(0) # d8=0
d.append(0) # d9=0

a = (d[1]*2 + d[3]*3 + d[5] + d[6] + d[7] + d[8] + d[9]*3 )
print('Test acc:', np.sum(np.argmax(a, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
