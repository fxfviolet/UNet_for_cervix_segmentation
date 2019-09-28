from model import *
from get_data import *
from sklearn.model_selection import train_test_split


class train(object):
    def __init__(self):
        print('begin training')
        self.im_height = 512
        self.im_width = 512

    def model_train(self,path,train=True):
        X, y = get_data(path, self.im_height, self.im_width,train)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.06, random_state=2019)

        input_img = Input((self.im_height, self.im_width, 1), name='img')
        model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[my_iou_metric])
        model.summary()

        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
            ModelCheckpoint('model-unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]

        # if 'model-unet.ht':
        #     model.load_weights('model-unet.h5')

        results = model.fit(X_train, y_train, batch_size=4, epochs=300, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))
        return results


if __name__=='__main__':
    path_train = './input/train/'
    results = train().model_train(path=path_train,train=True)