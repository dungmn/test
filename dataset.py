import glob, os,sys
import random, pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

def getPathFiles(path):
    cato = glob.glob(path+ '/*')
    list_path_images = list(map(lambda x: glob.glob(x + '/*.jpg'),cato))
    return list(map(lambda x: x.split('/')[1],cato)), list_path_images

class Dataset():
    def __init__(self, path):
        self.path = path
        self.cato, self.path_files = getPathFiles(self.path)

    def createTrainTest(self, iddb, p):
        train = []
        test= []
        path = 'db/db{}'.format(iddb)
        if not os.path.exists(path):
            os.makedirs(path)

        for db in self.path_files:
            lenght = int(len(db)*p)
            temp = db.copy()
            random.shuffle(temp)
            with open(path + '/train.txt', 'a+') as f:
                for trn in temp[:lenght]:
                    f.write(trn + '\n')

            with open(path + '/test.txt', 'a+') as f:
                for tst in temp[lenght:]:
                    f.write(tst + '\n')

    def extract(self,filename, model):
        img = image.load_img(filename,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features

    def extractFeatures(self, typefeat):
        if typefeat.lower() == 'vgg16_fc2':
            model = VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
        for db in self.path_files:
            for f in db:
                dirDS, dirCato, filename = f.split('/')
                ft = self.extract(f,model)
                path = typefeat + '/' + dirCato + '/'
                if not os.path.isdir(path):
                    os.makedirs(path)
                pickle.dump(ft, open(path +filename + '.pkl','wb'))
if __name__ == '__main__':
    ds = Dataset(sys.argv[2])
    if sys.argv[1] == 'createDB':
        ds.createTrainTest(sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'extractfeat':
        ds.extractFeatures(sys.argv[3])
