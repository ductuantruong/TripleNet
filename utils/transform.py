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


class SubtractMean(object):
    # TODO: pytorch tensor
    def __init__(self, mean):
        self.mean = mean 

    def __call__(self, img):
        return img.astype(np.float32) - self.mean


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

