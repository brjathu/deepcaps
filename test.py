

def test(model_path_file, model_path_weight, data, args):

    (x_train, y_train), (x_test, y_test) = data

    m1 = imp.load_source('module.name', model_path_file)
    _, eval_model = m1.DeepCapsNet28(input_shape=x_test.shape[1:], n_class=10, routings=3)
    eval_model.load_weights(model_path_weight)
    a1, b1 = eval_model.predict(x_test)
    p1 = np.sum(np.argmax(a1, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    print('Test acc:', p1)
    return p1
