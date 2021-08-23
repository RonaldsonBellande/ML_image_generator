from header_inputs import *


class generate_images(object):
    def __init__(self):

        # Bash size
        self.batch_size = 64

        # Model creation
        self.model = None

        # Data collecting
        (self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()

        # Single channel for dataset for training
        self.width, self.height, self.channels = self.trainX.shape[1], self.trainX.shape[2], 1
    
        # Reshape dataset
        self.trainX = self.trainX.reshape((self.trainX.shape[0], self.width, self.height, self.channels))
        self.testX = self.testX.reshape((self.testX.shape[0], self.width, self.height, self.channels))

        # Encode
        self.trainY = to_categorical(self.trainY)
        self.testY = to_categorical(self.testY)

        # Scalling
        self.datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Scale images
        self.scale_images()

        # Generate model
        self.create_model()

        # Saves model
        


    
    # Scale image
    def scale_images(self):
        
        # Scale image
        train_iterator = datagen.flow(self.trainX, self.trainY, self.batch_size)
        test_iterator = datagen.flow(self.testX, self.testY, self.batch_size)
    

    
    # Model creation
    def create_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))


    
    # Model evaluation
    def model_evaluation(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit model with generator
        model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)

        # Evaluate model
        _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)



    



