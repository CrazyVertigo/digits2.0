import os
import sys
import numpy as np
import fileSystemUtils as fs
import cv2 as cv2
#import cv2.cv as cv
import lmdb
import scipy.io
import os
import numpy as np
from scipy import io
import lmdb
from read_img import read_img_cv2
NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'
#options
#CAFFE_ROOT = '../../'
CAFFE_ROOT = '/home/hdu/caffe/'
phase = 'train'

sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe

#source data directory
data_dir = CAFFE_ROOT+'data/fcn_label_full/'

#lmdb destination: in which directory to save lmdb
lmdb_dst   = data_dir + 'lmdb/'


def main(args):
    imgs_dir = data_dir + 'fig'
    gt_dir = data_dir + 'mat'
    paths_imgs = fs.gen_paths(imgs_dir, fs.filter_is_img)
    paths_gt = fs.gen_paths(gt_dir)
    paths_pairs = fs.fname_pairs(paths_imgs, paths_gt)    
    paths_imgs, paths_gt = map(list, zip(*paths_pairs))

    lm_img_dst = lmdb_dst + phase + '_img_lmdb'
    lm_gt_dst  = lmdb_dst + phase + '_gt_lmdb'
    if not os.path.exists(lm_img_dst):
        print 'lmdb dir not exists,make it'
        os.makedirs(lm_img_dst)
    if not os.path.exists(lm_gt_dst):
        print 'lmdb dir not exists,make it'
        os.makedirs(lm_gt_dst)

    size1 = imgs_to_lmdb(paths_imgs, lm_img_dst, CAFFE_ROOT = CAFFE_ROOT)
    size2 = matfiles_to_lmdb(paths_gt, lm_gt_dst, 'gt',CAFFE_ROOT = CAFFE_ROOT)
    dif = size1 - size2
    dif = dif.sum()
    if(dif != 0):
         print 'ERROR: img-gt size not match! diff:'+str(diff)
         return 1
    return 0

def imgs_to_lmdb(paths_src, path_dst, CAFFE_ROOT=None):
    '''
    Generate LMDB file from set of images
    '''
    import numpy as np
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0, CAFFE_ROOT + 'python')
    import caffe
    
    db = lmdb.open(path_dst, map_size=int(1e12))
    size = np.zeros([len(paths_src), 2])
    with db.begin(write=True) as in_txn:
        i = 1
        for idx, path_ in enumerate(paths_src):
            print str(i)+' of '+str(len(paths_src))+' ...'
            
            img = read_img_cv2(path_)
            size[i-1, :] = img.shape[1:]
            img_dat = caffe.io.array_to_datum(img)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
            i = i + 1
    db.close()
    return size

def matfiles_to_lmdb(paths_src, path_dst, fieldname,
                     CAFFE_ROOT=None,
                     lut=None):
    if CAFFE_ROOT is not None:
        import sys
        sys.path.insert(0,  os.path.join(CAFFE_ROOT, 'python'))
    import caffe
    db = lmdb.open(path_dst, map_size=int(1e12))
    size = np.zeros([len(paths_src), 2])
    with db.begin(write=True) as in_txn:
        i = 1
        for idx, path_ in enumerate(paths_src):
            print str(i)+' of '+str(len(paths_src))+' ...'
            
            content_field = io.loadmat(path_)[fieldname]
            #print content_field.shape
            content_field = np.expand_dims(content_field, axis=0)   ##########
            content_field = content_field.astype(float)
            
            if lut is not None:
                content_field = lut(content_field)
            size[i-1, :] = content_field.shape[1:]
            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())
            i = i + 1
    
    db.close()
    return size

if __name__ == '__main__':
     main(None)
     pass
