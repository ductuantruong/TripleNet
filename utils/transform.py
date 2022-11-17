from __future__ import print_function

import numpy as np 

# scipy.ndimage -> skimage -> cv2, 
# skimage is one or two orders of magnitude slower than cv2
import cv2
from PIL import Image

try:
    import torch
except ImportError:
    pass

import collections
import numbers
import types

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

InterpolationFlags = {'nearest':cv2.INTER_NEAREST, 'linear':cv2.INTER_LINEAR, 
                       'cubic':cv2.INTER_CUBIC, 'area':cv2.INTER_AREA, 
                       'lanczos':cv2.INTER_LANCZOS4}

BorderTypes = {'constant':cv2.BORDER_CONSTANT, 
               'replicate':cv2.BORDER_REPLICATE, 'nearest':cv2.BORDER_REPLICATE,
               'reflect':cv2.BORDER_REFLECT, 'mirror': cv2.BORDER_REFLECT,
               'wrap':cv2.BORDER_WRAP, 'reflect_101':cv2.BORDER_REFLECT_101,}



def _loguniform(interval, random_state=np.random):
    low, high = interval
    return np.exp(random_state.uniform(np.log(low), np.log(high)))


def _clamp(img, min=None, max=None, dtype='uint8'):
    if min is None and max is None:
        if dtype == 'uint8':
            min, max = 0, 255
        elif dtype == 'uint16':
            min, max = 0, 65535
        else:
            min, max = -np.inf, np.inf
    img = np.clip(img, min, max)
    return img.astype(dtype)


def _jaccard(boxes, rect):
    def _intersect(boxes, rect):
        lt = np.maximum(boxes[:, :2], rect[:2])
        rb = np.minimum(boxes[:, 2:], rect[2:])
        inter = np.clip(rb - lt, 0, None)
        return inter[:, 0] * inter[:, 1]

    inter = _intersect(boxes, rect)
    
    area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area2 = (rect[2] - rect[0]) * (rect[3] - rect[1])
    union = area1 + area2 - inter 
    
    jaccard  = inter / np.clip(union, 1e-10, None)
    coverage = inter / np.clip(area1, 1e-10, None)
    return jaccard, coverage, inter


def _coords_clamp(cds, shape, outside=None):
    w, h = shape[1] - 1, shape[0] - 1
    if outside == 'discard':
        cds_ = []
        for x, y in cds:
            x_ = x if 0 <= x <= w else np.sign(x) * np.inf
            y_ = y if 0 <= y <= h else np.sign(y) * np.inf
            cds_.append([x_, y_])
        return np.array(cds_, dtype=np.float32)
    else:
        return np.array([[np.clip(cd[0], 0, w), np.clip(cd[1], 0, h)] for cd in cds], dtype=np.float32)


def _to_bboxes(cds, img_shape=None):
    assert len(cds) % 4 == 0

    h, w = img_shape if img_shape is not None else (np.inf, np.inf)
    boxes = []
    cds = np.array(cds)
    for i in range(0, len(cds), 4):
        xmin = np.clip(cds[i:i+4, 0].min(), 0, w - 1)
        xmax = np.clip(cds[i:i+4, 0].max(), 0, w - 1)
        ymin = np.clip(cds[i:i+4, 1].min(), 0, h - 1)
        ymax = np.clip(cds[i:i+4, 1].max(), 0, h - 1)
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)


def _to_coords(boxes):
    cds = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box 
        cds += [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ]
    return np.array(cds)


# recursively reset transform's state
def transform_state(t, **kwargs):
    if callable(t):
        t_vars = vars(t)

        if 'random_state' in kwargs and 'random' in t_vars:
            t.__dict__['random'] = kwargs['random_state']

        support = ['fillval', 'anchor', 'prob', 'mean', 'std', 'outside']
        for arg in kwargs:
            if arg in t_vars and arg in support:
                t.__dict__[arg] = kwargs[arg]

        if 'mode' in kwargs and 'mode' in t_vars:
            t.__dict__['mode'] = kwargs['mode']
        if 'border' in kwargs and 'border' in t_vars:
            t.__dict__['border'] = BorderTypes.get(kwargs['border'], cv2.BORDER_REPLICATE)

        if 'transforms' in t_vars:
            t.__dict__['transforms'] = transforms_state(t.transforms, **kwargs)
    return t


def transforms_state(ts, **kwargs):
    assert isinstance(ts, collections.Sequence)

    transforms = []
    for t in ts:
        if isinstance(t, collections.Sequence):
            transforms.append(transforms_state(t, **kwargs))
        else:
            transforms.append(transform_state(t, **kwargs))
    return transforms

class Unsqueeze(object):
    def __call__(self, img):
        if img.ndim == 2:
            return img[..., np.newaxis]
        elif img.ndim == 3:
            return img
        else:
            raise ValueError('input muse be image')



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 

    def __call__(self, img):
        # normalize np.ndarray or torch.FloatTensor
        if isinstance(img, np.ndarray):
            return (img - self.mean) / self.std
        elif isinstance(img, torch.FloatTensor):
            tensor = img
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
                return tensor 
        else:
            raise Exception('invalid input type')


class SubtractMean(object):
    # TODO: pytorch tensor
    def __init__(self, mean):
        self.mean = mean 

    def __call__(self, img):
        return img.astype(np.float32) - self.mean

class DivideBy(object):
    # TODO: pytorch tensor
    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, img):
        return img.astype(np.float32) / self.divisor


def HalfBlood(img, anchor, f1, f2):
    assert isinstance(f1, types.LambdaType) and isinstance(f2, types.LambdaType)

    if isinstance(anchor, numbers.Number):
        anchor = int(np.ceil(anchor))

    if isinstance(anchor, int) and img.ndim == 3 and 0 < anchor < img.shape[2]:
        img1, img2 = img[:,:,:anchor], img[:,:,anchor:]

        if img1.shape[2] == 1:
            img1 = img1[:, :, 0]
        if img2.shape[2] == 1:
            img2 = img2[:, :, 0]

        img1 = f1(img1)
        img2 = f2(img2)

        if img1.ndim == 2:
            img1 = img1[..., np.newaxis]
        if img2.ndim == 2:
            img2 = img2[..., np.newaxis]
        
        return np.concatenate((img1, img2), axis=2)
    elif anchor == 0:
        img = f2(img)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        return img
    else:
        img = f1(img)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        return img

# Photometric Transform
class RGB2BGR(object):
    def __call__(self, img):
        assert img.ndim == 3 and img.shape[2] == 3
        return img[:, :, ::-1]

class BGR2RGB(object):
    def __call__(self, img):
        assert img.ndim == 3 and img.shape[2] == 3
        return img[:, :, ::-1]


class GrayScale(object):
    # RGB to Gray
    def __call__(self, img):
        if img.ndim == 3 and img.shape[2] == 1:
            return img

        assert img.ndim == 3 and img.shape[2] == 3
        dtype = img.dtype
        gray = np.sum(img * [0.299, 0.587, 0.114], axis=2).astype(dtype)  #5x slower than cv2.cvtColor 
        
        #gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        return gray[..., np.newaxis]


class Hue(object):
    # skimage.color.rgb2hsv/hsv2rgb is almost 100x slower than cv2.cvtColor
    def __init__(self, var=0.05, prob=0.5, random_state=np.random):
        self.var = var
        self.prob = prob
        self.random = random_state

    def __call__(self, img):
        assert img.ndim == 3 and img.shape[2] == 3

        if self.random.random_sample() >= self.prob:
            return img

        var = self.random.uniform(-self.var, self.var)

        to_HSV, from_HSV = [(cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
                            (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)][self.random.randint(2)]

        hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

        hue = hsv[:, :, 0] / 179. + var
        hue = hue - np.floor(hue)
        hsv[:, :, 0] = hue * 179.

        img = cv2.cvtColor(hsv.astype('uint8'), from_HSV)
        return img


class Saturation(object):
    def __init__(self, var=0.3, prob=0.5, random_state=np.random):
        self.var = var 
        self.prob = prob
        self.random = random_state

        self.grayscale = GrayScale()

    def __call__(self, img):
        if self.random.random_sample() >= self.prob:
            return img

        dtype = img.dtype
        gs = self.grayscale(img)

        alpha = 1.0 + self.random.uniform(-self.var, self.var)
        img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
        return _clamp(img, dtype=dtype)



class Brightness(object):
    def __init__(self, delta=32, prob=0.5, random_state=np.random):
        self.delta = delta
        self.prob = prob
        self.random = random_state

    def __call__(self, img):
        if self.random.random_sample() >= self.prob:
            return img

        dtype = img.dtype
        #alpha = 1.0 + self.random.uniform(-self.var, self.var)
        #img = alpha * img.astype(np.float32)
        img = img.astype(np.float32) + self.random.uniform(-self.delta, self.delta)
        return _clamp(img, dtype=dtype)



class Contrast(object):
    def __init__(self, var=0.3, prob=0.5, random_state=np.random):
        self.var = var 
        self.prob = prob
        self.random = random_state

        self.grayscale = GrayScale()

    def __call__(self, img):
        if self.random.random_sample() >= self.prob:
            return img

        dtype = img.dtype
        gs = self.grayscale(img).mean()

        alpha = 1.0 + self.random.uniform(-self.var, self.var)
        img = alpha * img.astype(np.float32) + (1 - alpha) * gs
        return _clamp(img, dtype=dtype)


class RandomOrder(object):
    def __init__(self, transforms, random_state=None):  #, **kwargs):
        if random_state is None:
            self.random = np.random
        else:
            self.random = random_state
            #kwargs['random_state'] = random_state

        self.transforms = transforms_state(transforms, random=random_state)

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = self.random.permutation(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):
    def __init__(self, brightness=32, contrast=0.5, saturation=0.5, hue=0.1,
                     prob=0.5, random_state=np.random):
        self.transforms = []

        self.random = random_state
        if brightness != 0:
            self.transforms.append(
                Brightness(brightness, prob=prob, random_state=random_state))
        if contrast != 0:
            self.transforms.append(
                Contrast(contrast, prob=prob, random_state=random_state))
        if saturation != 0:
            self.transforms.append(
                Saturation(saturation, prob=prob, random_state=random_state))
        if hue != 0:
            self.transforms.append(
                Hue(hue, prob=prob, random_state=random_state))



# "ImageNet Classification with Deep Convolutional Neural Networks"
# looks inferior to ColorJitter
class FancyPCA(object):
    def __init__(self, var=0.2, random_state=np.random):
        self.var = var
        self.random = random_state

        self.pca = None    # shape (channels, channels)

    def __call__(self, img):
        dtype = img.dtype
        channels = img.shape[2]
        alpha = self.random.randn(channels) * self.var

        if self.pca is None:
            pca = self._pca(img)
        else:
            pca = self.pca

        img = img + (pca * alpha).sum(axis=1)
        return _clamp(img, dtype=dtype)

    def _pca(self, img):   # single image (hwc), or a batch (nhwc)
        assert img.ndim >= 3
        channels = img.shape[-1]
        X = img.reshape(-1, channels)

        cov = np.cov(X.T)   
        evals, evecs = np.linalg.eigh(cov)
        pca = np.sqrt(evals) * evecs
        return pca

    def fit(self, imgs):   # training
        self.pca = self._pca(imgs)
        print(self.pca)


def _expand(img, size, lt, val):
    h, w = img.shape[:2]
    nw, nh = size 
    x1, y1 = lt 
    expand = np.zeros([nh, nw] + list(img.shape[2:]), dtype=img.dtype)
    expand[...] = val
    expand[y1: h + y1, x1: w + x1] = img
    #expand = cv2.copyMakeBorder(img, y1, nh-h-y1, x1, nw-w-x1, 
    #							cv2.BORDER_CONSTANT, value=val)  # slightly faster
    return expand


class Pad(object):
    def __init__(self, padding, fillval=0, anchor=None):
        if isinstance(padding, numbers.Number):
            padding = (padding, padding)
        assert len(padding) == 2

        self.padding = [int(np.clip(_), 0, None) for _ in padding]
        self.fillval = fillval
        self.anchor = anchor

    def __call__(self, img, cds=None):
        if max(self.padding) == 0:
            return img if cds is None else (img, cds)

        h, w = img.shape[:2]
        pw, ph = self.padding

        pad = lambda im: _expand(im, (w + pw*2, h + ph*2), (pw, ph), self.fillval)
        purer = lambda im: _expand(im, (w + pw*2, h + ph*2), (pw, ph), 0)  
        img = HalfBlood(img, self.anchor, pad, purer)

        if cds is not None:
            return img, np.array([[x + pw, y + ph] for x, y in cds])
        else:
            return img


# "SSD: Single Shot MultiBox Detector".  generate multi-resolution image/ multi-scale objects
class Expand(object):
    def __init__(self, scale_range=(1, 4), fillval=0, prob=1.0, anchor=None, random_state=np.random):
        if isinstance(scale_range, numbers.Number):
            scale_range = (1, scale_range)
        assert max(scale_range) <= 5 

        self.scale_range = scale_range	
        self.fillval = fillval
        self.prob = prob
        self.anchor = anchor
        self.random = random_state

    def __call__(self, img, cds=None):
        if self.prob < 1 and self.random.random_sample() >= self.prob:
            return img if cds is None else (img, cds)

        #multiple = _loguniform(self.scale_range, self.random)
        multiple = self.random.uniform(*self.scale_range)

        h, w = img.shape[:2]
        nh, nw = int(multiple * h), int(multiple * w)

        if multiple < 1:
            return RandomCrop(size=(nw, nh), random_state=self.random)(img, cds)

        y1 = self.random.randint(0, nh - h + 1)
        x1 = self.random.randint(0, nw - w + 1)

        expand = lambda im: _expand(im, (nw, nh), (x1, y1), self.fillval)
        purer = lambda im: _expand(im, (nw, nh), (x1, y1), 0)
        img = HalfBlood(img, self.anchor, expand, purer)

        if cds is not None:
            return img, np.array([[x + x1, y + y1] for x, y in cds])
        else:
            return img


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = size

    def __call__(self, img, cds=None):
        h, w = img.shape[:2]
        tw, th = self.size

        if h == th and w == tw:
            return img if cds is None else (img, cds)
        elif h < th or w < tw:
            raise Exception('invalid crop size')

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img[y1:y1 + th, x1:x1 + tw]

        if cds is not None:
            return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
        else:
            return img


class RandomCrop(object):
    def __init__(self, size, fillval=0, random_state=np.random):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = size
        self.random = random_state

    def __call__(self, img, cds=None):
        h, w = img.shape[:2]
        tw, th = self.size

        assert h >= th and w >= tw

        x1 = self.random.randint(0, w - tw + 1)
        y1 = self.random.randint(0, h - th + 1)
        img = img[y1:y1 + th, x1:x1 + tw]

        if cds is not None:
            return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
        else:
            return img


class ObjectRandomCrop(object):
    def __init__(self, prob=1., random_state=np.random):
        self.prob = prob
        self.random = random_state 

        self.options = [
        #(0, None), 
        (0.1, None),     
        (0.3, None),
        (0.5, None),
        (0.7, None),
        (0.9, None),       
        (None, 1), ]
    

    def __call__(self, img, cbs, seg):
        h, w = img.shape[:2]

        if len(cbs) == 0:
            return img, cbs, seg

        # ad-hoc
        if len(cbs[0]) == 4:
            boxes = cbs
        elif len(cbs[0]) == 2:
            boxes = _to_bboxes(cbs, img.shape[:2])
        else:
            raise Exception('invalid input')

        params = [(np.array([0, 0, w, h]), None)]

        for min_iou, max_iou in self.options:
            if min_iou is None:
                min_iou = 0
            if max_iou is None:
                max_iou = 1

            for _ in range(50):
                scale = self.random.uniform(0.3, 1)
                aspect_ratio = self.random.uniform(
                    max(1 / 2., scale * scale),
                    min(2., 1 / (scale * scale)))
                th = int(h * scale / np.sqrt(aspect_ratio))
                tw = int(w * scale * np.sqrt(aspect_ratio))

                x1 = self.random.randint(0, w - tw + 1)
                y1 = self.random.randint(0, h - th + 1)
                rect = np.array([x1, y1, x1 + tw, y1 + th])

                iou, coverage, _ = _jaccard(boxes, rect)

                #m1 = coverage > 0.1
                #m2 = coverage < 0.45
                #if (m1 * m2).any():
                #	continue

                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(rect[:2] <= center, center < rect[2:]).all(axis=1)

                #mask = coverage >= 0.45
                #mask
                if not mask.any():
                    continue

                if min_iou <= iou.max() and iou.min() <= max_iou:
                    params.append((rect, mask))
                    break
        rect, mask = params[self.random.randint(len(params))]

        img = img[rect[1]:rect[3], rect[0]:rect[2]]
        seg = seg[rect[1]:rect[3], rect[0]:rect[2]]

        boxes[:, :2] = np.clip(boxes[:, :2], rect[:2], rect[2:])
        boxes[:, :2] = boxes[:, :2] - rect[:2]
        boxes[:, 2:] = np.clip(boxes[:, 2:], rect[:2], rect[2:])
        boxes[:, 2:] = boxes[:, 2:] - rect[:2]
        if mask is not None:
            boxes[np.logical_not(mask), :] = 0

        if len(cbs[0]) == 4:
            return img, boxes, seg
        else:
            return img, _to_coords(boxes), seg



class GridCrop(object):
    def __init__(self, size, grid=5, random_state=np.random):
        # 4 grids, 5 grids or 9 grids
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = size

        self.grid = grid
        self.random = random_state
        self.map = {
            0: lambda w, h, tw, th: (            0,            0),
            1: lambda w, h, tw, th: (       w - tw,            0),
            2: lambda w, h, tw, th: (       w - tw,       h - th),
            3: lambda w, h, tw, th: (            0,       h - th),
            4: lambda w, h, tw, th: ((w - tw) // 2, (h - th) // 2),
            5: lambda w, h, tw, th: ((w - tw) // 2,            0),
            6: lambda w, h, tw, th: (       w - tw, (h - th) // 2),
            7: lambda w, h, tw, th: ((w - tw) // 2,       h - th),
            8: lambda w, h, tw, th: (            0, (h - th) // 2),
        }

    def __call__(self, img, cds=None, index=None):
        h, w = img.shape[:2]
        tw, th = self.size
        if index is None:
            index = self.random.randint(0, self.grid)
        if index not in self.map:
            raise Exception('invalid index')

        x1, y1 = self.map[index](w, h, tw, th)
        img = img[y1:y1 + th, x1:x1 + tw]

        if cds is not None:
            return img, _coords_clamp([[x - x1, y - y1] for x, y in cds], img.shape)
        else:
            return img



class Resize(object):
    def __init__(self, size, mode='linear', anchor=None, random_state=np.random):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size, InterpolationMode.NEAREST),
            T.ToTensor()
        ]) 

    def __call__(self, img, cds=None, seg=None):

        h, w, c = img.shape
        tw, th = self.size

        #img = self.transform(img).numpy().transpose(2, 1, 0)
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        if cds is not None and seg is not None:
            s_x = tw / float(w)
            s_y = th / float(h)
            #seg = self.transform(seg).numpy().transpose(2, 1, 0)
            seg = cv2.resize(seg, dsize=self.size, interpolation=cv2.INTER_NEAREST)
            return img, np.array([[s_x * x, s_y * y] for x, y in cds]), seg
        else:
            return img

class HorizontalFlip(object):
    def __init__(self, prob=0.5, random_state=np.random):
        self.prob = prob
        self.random = random_state

    def __call__(self, img, cds=None, seg=None, flip=None):
        if flip is None:
            flip = self.random.random_sample() < self.prob

        if flip:
            img = img[:, ::-1]

        if cds is not None and seg is not None:
            if flip:
                seg = seg[:, ::-1]
            h, w = img.shape[:2]
            t = lambda x, y: [w-1-x, y] if flip else [x, y]
            return img, np.array([t(x, y) for x, y in cds]), seg
        else:
            return img


class VerticalFlip(object):
    def __init__(self, prob=0.5, random_state=np.random):
        self.prob = prob
        self.random = random_state

    def __call__(self, img, cds=None, flip=None):
        if flip is None:
            flip = self.random.random_sample() < self.prob

        if flip:
            img = img[::-1, :]
        
        if cds is not None:
            h, w = img.shape[:2]
            t = lambda x, y: [x, h-1-y] if flip else [x, y]
            return img, np.array([t(x, y) for x, y in cds])
        else:
            return img

# Pipeline
class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd 

    def __call__(self, *args):
        return self.lambd(*args)


class Merge(object):
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, *imgs):
        # ad-hoc 
        if len(imgs) > 1 and not isinstance(imgs[0], collections.Sequence):
            pass
        elif len(imgs) == 1 and isinstance(imgs[0], collections.Sequence):   # unreliable
            imgs = imgs[0]
        elif len(imgs) == 1:
            return imgs[0]
        else:
            raise Exception('input must be a sequence (list, tuple, etc.)')

        assert len(imgs) > 0 and all([isinstance(_, np.ndarray)
                    for _ in imgs]), 'only support numpy array'

        shapes = []
        imgs_ = []
        for i, img in enumerate(imgs):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=self.axis)
            imgs_.append(img)
            shape = list(img.shape)
            shape[self.axis] = None
            shapes.append(shape)
        assert all([_ == shapes[0] for _ in shapes]), 'shapes must match'
        return np.concatenate(imgs_, axis=self.axis)


class Split(object):
    def __init__(self, *slices, **kwargs):
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
            assert all([isinstance(s, slice) for s in slices_]), 'slices must consist of slice instances'

        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            result = []
            for s in self.slices:
                sl = [slice(None)] * img.ndim 
                sl[self.axis] = s 
                result.append(img[sl])
            return result
        else:
            raise Exception('object must be a numpy array')


class Compose(object):
    def __init__(self, transforms, random_state=None, **kwargs):
        if random_state is not None:
            kwargs['random_state'] = random_state
        self.transforms = transforms_state(transforms, **kwargs)

    def __call__(self, *data):
        # ad-hoc 
        if len(data) >= 1 and not isinstance(data[0], collections.Sequence):
            pass
        elif len(data) == 1 and isinstance(data[0], collections.Sequence) and len(data[0]) > 0:   # unreliable
            data = list(data[0])
        else:
            raise Exception('invalid input')

        for t in self.transforms:
            if not isinstance(data, collections.Sequence):   # unreliable
                data = [data]

            if isinstance(t, collections.Sequence):
                if len(t) > 1:
                    assert isinstance(data, collections.Sequence) and len(data) == len(t)
                    ds = []
                    for i, d in enumerate(data):
                        if callable(t[i]):
                            ds.append(t[i](d))
                        else:
                            ds.append(d)
                    data = ds
                elif len(t) == 1:
                    if callable(t[0]):
                        data = [t[0](data[0])] + list(data)[1:]
            elif callable(t):
                data = t(*data)
            elif t is not None:
                raise Exception('invalid transform type')

        if isinstance(data, collections.Sequence) and len(data) == 1:   # unreliable
            return data[0]
        else:
            return data

    def set_random_state(self, random_state):
        self.transforms = transforms_state(self.transforms, random=random_state)


class RandomCompose(Compose):
    def __init__(self, transforms, random_state=None, **kwargs):
        if random_state is None:
            random_state = np.random
        else:
            kwargs['random_state'] = random_state

        self.transforms = transforms_state(transforms, **kwargs)
        self.random = random_state

    def __call__(self, *data):
        self.random.shuffle(self.transforms)

        return super(RandomCompose, self).__call__(*data)


class ToNumpy(object):
    def __call__(self, pic):
        # torch.FloatTensor -> np.ndarray
        # or PIL Image -> np.ndarray
        # TODO
        pass


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # np.ndarray -> torch.FloatTensor
        # or PIL Image -> torch.FloatTensor

        # TODO: add option to choose whether div(255)
        if isinstance(img, np.ndarray):
            img = img.transpose((2, 0, 1))
            return torch.from_numpy(np.ascontiguousarray(img)).float()
        else:
            # TODO
            pass

class ToLongTensor(object):
    def __init__(self):
        pass 
    
    def __call_(self, label):
        if isinstance(label, np.ndarray):
            return torch.from_numpy(label).long()


class BoxesToCoords(object):
    def __init__(self, relative=False):
        self.relative = relative

    def bbox2coords(self, bbox):
        xmin, ymin, xmax, ymax = bbox 
        return np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ])

    def __call__(self, img, boxes, seg):
        if len(boxes) == 0:
            return img, np.array([])

        h, w = img.shape[:2]
        if self.relative:
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
        return img, np.vstack([self.bbox2coords(_) for _ in boxes]), seg


class CoordsToBoxes(object):
    def __init__(self, relative=True):
        self.relative = relative

    def coords2bbox(self, cds, w, h):
        xmin = np.clip(cds[:, 0].min(), 0, w - 1)
        xmax = np.clip(cds[:, 0].max(), 0, w - 1)
        ymin = np.clip(cds[:, 1].min(), 0, h - 1)
        ymax = np.clip(cds[:, 1].max(), 0, h - 1)
        return np.array([xmin, ymin, xmax, ymax])

    def __call__(self, img, cds, seg):
        if len(cds) == 0:
            return img, np.array([])

        assert len(cds) % 4 == 0
        num = len(cds) // 4

        h, w = img.shape[:2]
        boxcds = np.split(np.array(cds), np.arange(1, num) * 4)
        boxes = np.array([self.coords2bbox(_, w, h) for _ in boxcds])

        if self.relative:
            boxes[:, 0] /= float(w) 
            boxes[:, 2] /= float(w)
            boxes[:, 1] /= float(h)
            boxes[:, 3] /= float(h)

        return img, boxes, seg

