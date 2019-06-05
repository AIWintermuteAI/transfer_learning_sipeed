import keras
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
import argparse
import os

def prepare_image(file, show=False, predictions=None):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if show:
        plt.imshow(img)   
        plt.text(0.2, 0.2, predictions, bbox=dict(facecolor='red', alpha=0.5))                        
        plt.axis('off')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


parser = argparse.ArgumentParser()
parser.add_argument('--mode',dest='mode')
parser.add_argument('--dataset_location', dest='dataset_location')
parser.add_argument('--number_of_classes',dest='number_of_classes',type=int)
parser.add_argument('--test', dest='test', default=False, action='store_true')
parser.add_argument('--test_location', dest='test_location')
args = parser.parse_args()
	
base_model=keras.applications.mobilenet.MobileNet(input_shape=(128, 128, 3), alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", classes = 1000)


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(100,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dropout(0.5)(x)
x=Dense(50,activation='relu')(x) #dense layer 3
preds=Dense(args.number_of_classes,activation='softmax')(x) #final layer with softmax activation


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

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1) #included in our dependencies

train_generator=train_datagen.flow_from_directory(args.dataset_location,
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical', 
						 shuffle=True,   							 							 subset='training')

validation_generator=train_datagen.flow_from_directory(args.dataset_location,
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical', 
						 shuffle=True,
						 subset='validation')
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
fo = open("labels_101.txt", "w")
for k,v in labels.items():
    print(v)
    fo.write(v+"\n")
fo.close()
model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer, loss function will be categorical cross entropy, evaluation metric will be accuracy
if args.mode == 'train':
    print("Train mode")
    step_size_train=train_generator.n//train_generator.batch_size
    step_size_validation=train_generator.n//validation_generator.batch_size
    model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=10)
    print("Saving model")
    model.save(args.dataset_location+'_model.h5')
else:
    print("Load model mode")
    model.load_weights(args.dataset_location+'_model.h5')

if args.test == True:
    print("Testing on the images model hasn't seen in training")
    for filename in os.listdir(args.test_location):
        preprocessed_image = prepare_image(os.path.join(args.test_location,filename),show=False)
        pred = model.predict(preprocessed_image)
        predicted_class_indices=np.argmax(pred,axis=1)
        predictions = [labels[k] for k in predicted_class_indices]
        print(predictions)
        preprocessed_image = prepare_image(os.path.join(args.test_location,filename),show=True, predictions=predictions)





