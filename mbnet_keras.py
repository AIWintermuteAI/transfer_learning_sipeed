import keras
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from mobilenet_sipeed.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


base_model=MobileNet(input_shape=(128, 128, 3), alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", classes = 1000, backend=keras.backend, layers=keras.layers,models=keras.models,utils=keras.utils)


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(100,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dropout(0.5)(x)
x=Dense(50,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


for i,layer in enumerate(model.layers):
    print(i,layer.name)

    # or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('images',
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical', shuffle=True)
model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer, loss function will be categorical cross entropy, evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=10)

model.save('my_model.h5')

#model.load_weights('my_model.h5')







