import glob
import os
import numpy as np

IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']

if __name__ == '__main__':
    np.random.seed(0)
    cls_num = 500
    ratio = 0.5
    fout_train_label = open('ImageNet1000_label_%d_%.2f.txt' % (cls_num, ratio), 'w')
    fout_train_unlabel = open('ImageNet1000_unlabel_%d_%.2f.txt' % (cls_num, ratio), 'w')

    class_list = os.listdir('/home/sunyiyou/dataset/imagenet/val')
    np.random.shuffle(class_list)
    for i, folder_name in enumerate(class_list):
        files = []
        for extension in IMG_EXTENSIONS:
            files.extend(glob.glob(os.path.join('/home/sunyiyou/dataset/ILSVRC-2012/train', folder_name, '*' + extension) ))
        for filename in files:
            if i < cls_num and np.random.rand() < ratio:
                fout_train_label.write('%s %d\n'%(filename[41:], i))
            else:
                fout_train_unlabel.write('%s %d\n'%(filename[41:], i))

    fout_train_label.close()
    fout_train_unlabel.close()