from model import *
from get_data import *
import tensorflow as tf


class predict(object):
    def __init__(self):
        print('begin predicting')
        self.im_height = 512
        self.im_width = 512

    def get_iou(self, A, B):
        batch_size = A.shape[0]
        metric = []
        for batch in range(batch_size):
            t, p = A[batch] > 0, B[batch] > 0
            intersection = np.logical_and(t, p)
            union = np.logical_or(t, p)
            iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
            if iou >= 0.5:
                metric.append(1)
            else:
                metric.append(0)
        return np.sum(metric,dtype=np.float64)

    def get_iou_metric(self,label, preds_test):
        return tf.py_func(self.get_iou, [label, preds_test > 0.5], tf.float64)

    def predict_accuracy(self,path,train=True):
        X_test, y_test = get_data(path, self.im_height,self.im_width, train)
        input_img = Input((self.im_height, self.im_width, 1), name='img')
        model = get_unet(input_img, n_filters=16, batchnorm=True)
        model.load_weights('model-unet.h5')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        size = 20
        total_accuracy = 0
        for i in range(0, len(X_test), size):
            print('i=', i)
            label = y_test[i:i+size]
            preds_test = model.predict(X_test[i:i + size])
            accuracy = self.get_iou_metric(label, preds_test)
            total_accuracy += accuracy
        ave_accuracy = sess.run(total_accuracy / len(X_test))
        return ave_accuracy


if __name__=='__main__':
    path_test = './input/test/'
    ave_accuracy = predict().predict_accuracy(path=path_test,train=True)
    print('ave_accuracy = ',ave_accuracy)
