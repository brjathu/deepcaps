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
x_test64 = np.load("../../x_test.npy")


# input_shape=x_train.shape[1:]
# model_input = Input(shape=input_shape)
# ensemble_model = ensemble(models, model_input)

# y_pred = ensemble_model.predict([x_test]*len(models), batch_size=50)
# print('-' * 30 + 'Begin: test' + '-' * 30)
# print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

# # majority voting
# avg_ensamble_model = VotingModel(models)
# y_pred = avg_ensamble_model.predict(x_test, batch_size=200)
# y_pred = np.reshape(y_pred, (-1,1))
# yy = np.array([i[0] for i in y_pred])
# print('Test acc:', np.sum(yy == np.argmax(y_test, 1)) / y_test.shape[0])







m1 = imp.load_source('module.name', 'model/CIFAR10_ensemble/1/deepcaps.py')
_, eval_model1 = m1.DeepCapsNet(input_shape=x_test64.shape[1:], n_class=10, routings=3)
eval_model1.load_weights('model/CIFAR10_ensemble/1/best_weights.h5')
a1, b1 = eval_model1.predict(x_test64)
np.save("model/CIFAR10_ensemble/ensamble/deepcaps_1.npy", a1)


m1 = imp.load_source('module.name', 'model/CIFAR10_ensemble/2/deepcaps.py')
_, eval_model1 = m1.DeepCapsNet(input_shape=x_test64.shape[1:], n_class=10, routings=3)
eval_model1.load_weights('model/CIFAR10_ensemble/2/best_weights.h5')
a2, b1 = eval_model1.predict(x_test64)
np.save("model/CIFAR10_ensemble/ensamble/deepcaps_2.npy", a2)

m1 = imp.load_source('module.name', 'model/CIFAR10_ensemble/3/deepcaps.py')
_, eval_model1 = m1.DeepCapsNet(input_shape=x_test64.shape[1:], n_class=10, routings=3)
eval_model1.load_weights('model/CIFAR10_ensemble/3/best_weights.h5')
a3, b1 = eval_model1.predict(x_test64)
np.save("model/CIFAR10_ensemble/ensamble/deepcaps_3.npy", a3)


m1 = imp.load_source('module.name', 'model/CIFAR10_ensemble/4/deepcaps.py')
_, eval_model1 = m1.DeepCapsNet(input_shape=x_test64.shape[1:], n_class=10, routings=3)
eval_model1.load_weights('model/CIFAR10_ensemble/4/best_weights.h5')
a4, b1 = eval_model1.predict(x_test64)
np.save("model/CIFAR10_ensemble/ensamble/deepcaps_4.npy", a4)





d1 = np.load("model/CIFAR10_ensemble/ensamble/deepcaps_1.npy")
p1 = np.sum(np.argmax(d1, 1) == t) / y_test.shape[0]
print('Test acc:', p1)

d2 = np.load("model/CIFAR10_ensemble/ensamble/deepcaps_2.npy")
p2 = np.sum(np.argmax(d2, 1) == t) / y_test.shape[0]
print('Test acc:', p2)

d3 = np.load("model/CIFAR10_ensemble/ensamble/deepcaps_3.npy")
p3 = np.sum(np.argmax(d3, 1) == t) / y_test.shape[0]
print('Test acc:', p3)

d4 = np.load("model/CIFAR10_ensemble/ensamble/deepcaps_4.npy")
p4 = np.sum(np.argmax(d4, 1) == t) / y_test.shape[0]
print('Test acc:', p4)


a = (d1 + d2 + d3 + d4)
print('Ensemble Test acc:', np.sum(np.argmax(a, 1) == np.argmax(y_test, 1)) / y_test.shape[0])