from keras import optimizers
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from load_datasets import *
from utils import margin_loss, CustomModelCheckpoint
from deepcaps import DeepCapsNet, BaseCapsNet


def train(model, data, args):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log' + appendix + '.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                              batch_size=args.batch_size, histogram_freq=int(args.debug),write_grads=False)
    checkpoint1 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_1' + appendix + '.h5', monitor='val_capsnet_acc',
                                         save_best_only=False, save_weights_only=True, verbose=1)

    checkpoint2 = CustomModelCheckpoint(model, args.save_dir + '/best_weights_2' + appendix + '.h5', monitor='val_capsnet_acc',
                                         save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * 0.5**(epoch // 10))
    
    if(args.numGPU>1):
        parallel_model = multi_gpu_model(model, gpus=args.numGPU)
    else:
        parallel_model = model

    # compile the model  ssim loss for better reconstruction DSSIMObjective()
    parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                           loss=[margin_loss, 'mse'],
                           loss_weights=[1, 0.4],
                           metrics={'capsnet': "accuracy"})

    # Begin: Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=args.shift_fraction):
        train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                           samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.1,
                                           width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0,
                                           zoom_range=0.1, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                                           vertical_flip=False, rescale=None, preprocessing_function=None,
                                           data_format=None)  # shift up to 2 pixel for MNIST
        train_datagen.fit(x)
        generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=True)
        while True:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    parallel_model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                                 steps_per_epoch=int(y_train.shape[0] / args.batch_size), epochs=args.epochs,
                                 validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[lr_decay, log, checkpoint1, checkpoint2],
                                 initial_epoch=int(args.ep_num),
                                 shuffle=True)


    parallel_model.save(args.save_dir + '/trained_model_multi_gpu.h5')
    model.save(args.save_dir + '/trained_model.h5')

    return parallel_model




class args:
    numGPU = 3
    epochs = 100
    batch_size = 256
    lr = 0.001
    lr_decay = 0.96
    lam_recon = 0.4
    r = 3
    routings = 3
    shift_fraction = 0.1
    debug = False
    digit = 5
    save_dir = 'model/cifar_64'
    t = False
    w = None
    ep_num = 0


try:
    os.system("mkdir " + args.save_dir)
except:
    print("mkdir " + args.save_dir)
try:
    os.system("cp *.py " + args.save_dir + "/")
except:
    print("cp *.py " + args.save_dir + "/")
    
    
# load data
# (x_train, y_train), (x_test, y_test) = load_cifar100()
(x_train, y_train), (x_test, y_test) = load_cifar10()
# (x_train, y_train), (x_test, y_test) = load_svhn()
# x_train,y_train,x_test,y_test = load_tiny_imagenet("tiny_imagenet/tiny-imagenet-200", 200)

x_train = resize(x_train)
x_test = resize(x_test)

model, eval_model = DeepCapsNet(input_shape=x_train.shape[1:], n_class=y_train.shape[1], routings=args.routings)
# plot_model(model, show_shapes=True,to_file=args.save_dir + '/model.png')

appendix = ""

train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
