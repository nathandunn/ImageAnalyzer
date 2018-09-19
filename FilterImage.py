

print("running")


import keras
import keras.applications
# import keras_resnet.models
# import keras_resnet.layers
# shape, classes = (32, 32, 3), 10
# x = keras.layers.Input(shape)

# https://github.com/tensorflow/tensorflow/issues/20690

# model1 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#
# model = keras_resnet.models.ResNet50(x, classes=classes)
# model.compile("adam", "categorical_crossentropy", ["accuracy"])
# (training_x, training_y), (_, _) = keras.datasets.cifar10.load_data()
# training_y = keras.utils.np_utils.to_categorical(training_y)
# model.fit(training_x, training_y)


# #-------------------------------------
# #   Load pre-trained models
# #-------------------------------------
# resnet50  = resnet.ResNet50(weights='imagenet')
# resnet101 = resnet.ResNet101(weights='imagenet')
# resnet152 = resnet.ResNet152(weights='imagenet')
#
# #-------------------------------------
# #   Helper functions
# #-------------------------------------
# def path_to_tensor(image_path, target_size):
#     image = load_img(image_path, target_size=target_size)
#     tensor = img_to_array(image)
#     tensor = np.expand_dims(tensor, axis=0)
#     return tensor
#
# #-------------------------------------
# #   Make predictions
# #-------------------------------------
# image_path = 'examples/images/dog.jpeg'
# image_tensor = path_to_tensor(image_path, (224, 224))
# pred_resnet50  = np.argmax(resnet50.predict(image_tensor))
# pred_resnet101 = np.argmax(resnet101.predict(image_tensor))
# pred_resnet152 = np.argmax(resnet152.predict(image_tensor))

print("finished")