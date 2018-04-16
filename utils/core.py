import csv
from enum import IntEnum
import logging
import math
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from openslide import OpenSlide
from PIL import Image
from scipy import io as sio
from skimage import morphology
from skimage import transform
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters.rank import median


_SMALL_HOLE_AREA = 128
_SMALL_OBJECT_AREA = 128
_MEDIAN_DISK = 17
_ROI_PATCH_COVER_PERCENTAGE = 0.4


class Centre(IntEnum):
    CENTRE_0 = 0
    CENTRE_1 = 1
    CENTRE_2 = 2
    CENTRE_3 = 3
    CENTRE_4 = 4


class WSIData:

    def __init__(self, tif_path, centre, is_excluded, label_tif_path=None, label_xml_path=None):
        self._wsi_slide = OpenSlide(str(tif_path))
        self._tif_path = tif_path
        self._is_excluded = is_excluded
        self._label_tif_path = label_tif_path
        self._label_xml_path = label_xml_path
        self._label_slide = None

        if not isinstance(centre, Centre):
            raise TypeError('centre must be an instance of {}.Centre.'
                            .format(self.__class__.__name__))
        self._centre = centre
        self._name = tif_path.stem

    def _get_label_slide(self):
        if self._label_slide is None and self._label_tif_path is not None:
            self._label_slide = (self._label_tif_path and
                                 OpenSlide(str(self._label_tif_path)))
        return self._label_slide

    def _roi_threshold(self, img_np):
        '''Performs thresholding on the WSI image to extract the tissue region.

        RGB image is converted to HSV color space. The H and S channels are
        then thresholded via Otsu's Threshold, then combined via bitwise AND.
        Morphological transormation was applied by removing small holes and
        regions, and was filtered using median blur.

        Parameters
        ----------
        img_np: np.uint8[H, W, 3] np.array
            RGB WSI as an np.array image

        Returns
        -------
        bool[H, W] np.array
            ROI mask of the WSI.
        '''
        img_hsv = rgb2hsv(img_np)
        channel_h = img_hsv[:, :, 0]
        channel_s = img_hsv[:, :, 1]

        thresh_h = threshold_otsu(channel_h)
        thresh_s = threshold_otsu(channel_s)

        binary_h = channel_h > thresh_h
        binary_s = channel_s > thresh_s

        binary = np.bitwise_and(binary_h, binary_s)
        binary = morphology.remove_small_objects(binary, _SMALL_OBJECT_AREA)
        binary = morphology.remove_small_holes(binary, _SMALL_HOLE_AREA)
        binary = median(binary, morphology.disk(_MEDIAN_DISK))

        return binary.astype(bool)


    def _mark_metastases_regions_in_label(self, blank_mask_np, label_np):
        '''Marks the blank mask using the given label image.

        This is to be overriden due to the different nature of the labels
        from the CAMELYON 16 and 17 data set.

        Parameters
        ----------
        blank_mask_np: bool[H, W] np.array
            Mask to serve as the new label

        label_np: np.uint[H, W] np.array
            Gray scale image with the mask values.

        Returns
        -------
        bool[H, W] np.array
            Mask to serve as the new label
        '''
        blank_mask_np[np.where(label_np == 1)] = True
        blank_mask_np[np.where(label_np == 2)] = False


    @property
    def name(self):
        return self._name

    @property
    def centre(self):
        return self._centre

    @property
    def label_xml_path(self):
        return self._label_xml_path

    @property
    def tif_path(self):
        return self._tif_path

    @property
    def is_excluded(self):
        return self._is_excluded

    def get_full_wsi_image(self, level):
        '''Returns the whole WSI as an RGB image.

        Parameters
        ----------
        level: int
            Level downsample to read the WSI. (values: 0..8)

        Returns
        -------
        np.uint8[H, W, 3] np.array
            Whole WSI RGB image.
        '''
        wsi_dim = self._wsi_slide.level_dimensions[level]
        wsi_img = self._wsi_slide.read_region((0, 0), level, wsi_dim)
        return np.array(wsi_img.convert('RGB'))


    def get_level_dimension(self, level):
        '''Returns the dimensions of the WSI for a given level.

        Parameters
        ----------
        level: int
            Level downsample to get dimension. (values: 0..8)

        Returns
        -------
        (width: int, height: int)
            Dimension of the WSI.
        '''
        return self._wsi_slide.level_dimensions[level]


    def get_level_downsample(self, level):
        '''Returns the dimensions of the WSI for a given level.

        Parameters
        ----------
        level: int
            Level downsample to get dimension. (values: 0..8)

        Returns
        -------
        (width: int, height: int)
            Dimension of the WSI.
        '''
        return self._wsi_slide.level_downsamples[level]


    def get_roi_mask(self, level):
        '''Returns the ROI of the WSI i.e. the tissue region.

        Parameters
        ----------
        level: int
            Level downsample to read the WSI. (values: 0..8)

        Returns
        -------
        bool[H, W] np.array
            Tissue ROI mask of the WSI.
        '''
        wsi_img = self.get_full_wsi_image(level)
        wsi_img_np = np.array(wsi_img, dtype=np.uint8)
        roi_mask = self._roi_threshold(wsi_img_np)
        metastases_roi_mask, label_np = self._get_metastases_mask(level)
        roi_mask[np.where(metastases_roi_mask)] = True
        roi_mask[np.where(label_np == 3)] = False
        return roi_mask


    def _get_metastases_mask(self, level):
        '''Get metastases ROI of the WSI.

        Parameters
        ----------
        level: int
            Level downsample to read the WSI. (values: 0..8)

        Returns
        -------
        bool[H, W] np.array
            Metastases ROI mask of the WSI.
        '''
        wsi_w, wsi_h = self._wsi_slide.level_dimensions[level]
        metastases_mask = np.full((wsi_h, wsi_w), False)

        label_slide = self._get_label_slide()
        if label_slide is not None:
            label_dim = label_slide.level_dimensions[level]
            label_img = label_slide.read_region((0, 0), level, label_dim)
            label_np = np.array(label_img.convert('L'))
            label_w, label_h = label_dim
            label_np = label_np[:min(wsi_h, label_h), :min(wsi_w, label_w)]
            self._mark_metastases_regions_in_label(metastases_mask, label_np)

        return metastases_mask, label_np


    def get_metastases_mask(self, level):
        return self._get_metastases_mask(level)[0]


    def read_region_and_label(self, coordinates, level, dimension):
        '''Extracts a RGB region from the WSI.

        The given coordinates will be the center of the region, with the
        given dimension.

        Parameters
        ----------
        coordinates: (x: int, y: int)
            Coordinates in the WSI at level 0. Upper-left of the region.

        level: int
            Level downsample to read the WSI. (values: 0..8)

        dimension: (w: int, h: int)
            Dimension of the region to extract.

        Returns
        -------
        np.uint8[H, W, 3] np.array
            RGB region from WSI.

        bool[H, W] np.array
            Label mask for the given region.
        '''
        w, h = dimension
        x, y = coordinates

        ## coordinate is at the center of the patches
        # x, y = x - (w * 2**level // 2), y - (h * 2**level // 2)
        args = (x, y), level, dimension

        patch_img = self._wsi_slide.read_region(*args)
        patch_np = np.array(patch_img.convert('RGB'))
        metastases_np = np.full((w, h), False)

        label_slide = self._get_label_slide()
        if label_slide is not None:
            label_img = label_slide.read_region(*args)
            label_np = np.array(label_img.convert('L'))
            self._mark_metastases_regions_in_label(metastases_np, label_np)

        return patch_np, metastases_np

    def _get_roi_patch_positions(self, level, get_roi, stride=256,
                                 patch_side=513, ds_level=5):
        '''Positions returned are in level 0.'''
        assert ds_level > level, 'level must be less than ds_level'

        width, height = self._wsi_slide.level_dimensions[level]
        ds_roi = get_roi(ds_level)
        ds = 2**(ds_level - level)
        upscale_factor = 2**level

        for y in range(0, math.ceil(height / stride) * stride, stride):
            for x in range(0, math.ceil(width / stride) * stride, stride):
                y_ds = y // ds
                x_ds = x // ds
                y_ds_end = (y + stride) // ds
                x_ds_end = (x + stride) // ds
                roi_patch = ds_roi[y_ds:y_ds_end, x_ds:x_ds_end]

                if (np.sum(roi_patch) / roi_patch.size >
                    _ROI_PATCH_COVER_PERCENTAGE):
                    yield x * upscale_factor, y * upscale_factor


    def get_roi_patch_positions(self, level, stride=256, patch_side=513,
                                ds_level=5, force_not_excluded=False):
        if not self._is_excluded or force_not_excluded:
            get_roi = self.get_roi_mask
        else:
            get_roi = self.get_metastases_mask

        return self._get_roi_patch_positions(
            level,
            get_roi,
            stride,
            patch_side,
            ds_level,
        )

    def close(self):
        self._wsi_slide.close()
        self._label_slide is not None and self._label_slide.close()


    def __repr__(self):
        return '<Centre: {}, Name: {}>'.format(self._centre, self._name)


_CENTRE_STR_VALS = tuple(str(i) for i in range(5))
_BOOL_STR_VALS = ('0', '1')

def parse_dataset(filelist_path):
    '''Parse data from CSV file to a list of WSIData.

    The CSV file should contain this row format.
    tif_path, label_path, camelyon_{16,17}, centre

    Parameters
    ----------
    filelist_path: Path
        Path to the csv file.

    Returns
    -------
    WSIData[]
        A list of WSI data parsed from the file.
    '''
    wsi_data = []
    with open(str(filelist_path)) as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        for i, line in enumerate(csvreader):
            (tif_path, label_tif_path, label_xml_path, release_group,
             centre, is_excluded) = line
            tif_path = Path(tif_path)
            label_tif_path = (label_tif_path or None) and Path(label_tif_path)
            label_xml_path = (label_xml_path or None) and Path(label_xml_path)

            if centre not in _CENTRE_STR_VALS:
                raise ValueError(
                    'Skipping row {}; {} is not a valid centre; use {}'
                    .format(i, centre, _CENTRE_STR_VALS))

            if is_excluded not in _BOOL_STR_VALS:
                raise ValueError('Skipping row {}; {} not a 0 or 1'
                                 .format(i, is_excluded))

            kwargs = {
                'tif_path': tif_path,
                'centre': Centre(int(centre)),
                'label_tif_path': label_tif_path,
                'label_xml_path': label_xml_path,
                'is_excluded': bool(int(is_excluded)),
            }

            data = WSIData(**kwargs)

            wsi_data.append(data)
    return wsi_data


def save_patch(patch_np, save_path, img_format='JPEG'):
    img = Image.fromarray(patch_np)
    img.save(str(save_path), img_format)


def save_label(label_np, save_path):
    sio.savemat(str(save_path), {'data': label_np})


def get_logger(name):
    logfilename = name.lower().replace(' ', '_') + '.log'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def __test_relative_downsample_sizes():
    data_list_path_s = '/media/shishigami/6CC13AD35BD48D86/C16Data/train/data.csv'
    wsi_data = parse_dataset(data_list_path_s)
    slide = wsi_data[0]

    side = 513
    coord = 71880, 125850
    lvl = 1
    lvl_ds = 2
    ds = 2**(lvl_ds - lvl)
    img, _ = slide.read_region_and_label(coord, lvl, (side, side))
    img_ds, _ = slide.read_region_and_label(coord, lvl_ds, (int(side // ds), int(side // ds)))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(img_ds)
    plt.show()

def __test_get_roi_patch_positions():
    data_list_path_s = '/media/shishigami/6CC13AD35BD48D86/C16Data/train/data.csv'
    wsi_data = parse_dataset(data_list_path_s)
    slide = wsi_data[9]

    level = 2
    stride = 256
    patch_side = 513
    dim = patch_side, patch_side
    ds_level = 5
    w, h = slide.get_level_dimension(ds_level)
    test_mask = np.full((h, w), False)
    ds = 2**(ds_level - level)

    for x, y in slide.get_roi_patch_positions(level, stride, patch_side, ds_level):
        y_lvl = y // 2**level
        x_lvl = x // 2**level
        y_ds = y_lvl  // ds
        x_ds = x_lvl // ds
        y_ds_end = (y_lvl + patch_side) // ds
        x_ds_end = (x_lvl + patch_side) // ds
        test_mask[y_ds:y_ds_end, x_ds:x_ds_end] = True

    ds_mask = slide.get_roi_mask(ds_level)

    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
    axarr[0].imshow(ds_mask)
    axarr[1].imshow(test_mask)
    plt.show()

def __test_get_metastases_roi_patch_positions():
    data_list_path_s = '/media/shishigami/6CC13AD35BD48D86/C16Data/train/data_test_001.csv'
    wsi_data = parse_dataset(data_list_path_s)
    slide = wsi_data[1]
    print(slide.name)

    level = 2
    stride = 256
    patch_side = 513
    ds_level = 5
    w, h = slide.get_level_dimension(ds_level)
    test_mask = np.full((h, w), False)
    ds = 2**(ds_level - level)

    count = 0
    for x, y in slide.get_roi_patch_positions(level, stride, patch_side, ds_level, force_not_excluded=True):
        y_ds = y // ds
        x_ds = x // ds
        y_ds_end = (y + patch_side) // ds
        x_ds_end = (x + patch_side) // ds
        test_mask[y_ds:y_ds_end, x_ds:x_ds_end] = True
        count += 1
    print('{} patches extracted.'.format(count))

    ds_mask = slide.get_roi_mask(ds_level)

    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
    axarr[0].imshow(ds_mask)
    axarr[1].imshow(test_mask)
    plt.show()


if __name__ == '__main__':
    __test_get_roi_patch_positions()
