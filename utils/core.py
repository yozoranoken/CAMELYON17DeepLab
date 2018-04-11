from abc import ABC, abstractmethod
import csv
from enum import IntEnum
import logging
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from openslide import OpenSlide
from skimage import morphology
from skimage import transform
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters.rank import median


_SMALL_HOLE_AREA = 128
_SMALL_OBJECT_AREA = 128
_MEDIAN_DISK = 17


class Centre(IntEnum):
    CENTRE_0 = 0
    CENTRE_1 = 1
    CENTRE_2 = 2
    CENTRE_3 = 3
    CENTRE_4 = 4


class WSIData(ABC):

    def __init__(self, tif_path, centre, label_tif_path=None, label_xml_path=None):
        self._wsi_slide = OpenSlide(str(tif_path))
        self._tif_path = tif_path
        self._label_tif_path = label_tif_path
        self._label_xml_path = label_xml_path
        self._label_slide = None

        if not isinstance(centre, Centre):
            raise TypeError('centre must be an instance of {}.Centre.'
                            .format(self.__class__.__name__))
        self._centre = centre
        self._name = tif_path.stem

    def _load_label_slide(self):
        if self._label_slide is None and self._label_tif_path is not None:
            self._label_slide = (self._label_tif_path and
                                 OpenSlide(str(self._label_tif_path)))

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
        return binary

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
        return roi_mask


    @abstractmethod
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
        pass


    def get_metastases_mask(self, level):
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

        self._load_label_slide()
        if self._label_slide is not None:
            label_dim = self._label_slide.level_dimensions[level]
            label_img = self._label_slide.read_region((0, 0), level, label_dim)
            label_np = np.array(label_img.convert('L'))
            label_w, label_h = label_dim
            label_np = label_np[:min(wsi_h, label_h), :min(wsi_w, label_w)]
            self._mark_metastases_regions_in_label(metastases_mask, label_np)

        return metastases_mask


    def read_region_and_label(self, coordinates, level, dimension):
        '''Extracts a RGB region from the WSI.

        The given coordinates will be the center of the region, with the
        given dimension.

        Parameters
        ----------
        coordinates: (x: int, y: int)
            Coordinates in the WSI at level 0.

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
        cx, cy = x - (w * 2**level // 2), y - (h * 2**level // 2)
        args = (cx, cy), level, dimension
        patch_img = self._wsi_slide.read_region(*args)
        patch_np = np.array(patch_img.convert('RGB'))

        self._load_label_slide()
        if self._label_slide is not None:
            label_img = self._label_slide.read_region(*args)
            label_np = np.array(label_img.convert('L'))
            label_np = np.ma.masked_greater(label_np, 0).mask
        else:
            label_np = np.full((w, h), False)

        return patch_np, label_np


    def __repr__(self):
        return '<Centre: {}, Name: {}>'.format(self._centre, self._name)


class C16WSIData(WSIData):
    def _mark_metastases_regions_in_label(self, blank_mask_np, label_np):
        '''Marks the blank mask using the given label image.

        The images are simply labeled with 255 for positive and 0 for negative.

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


class C17WSIData(WSIData):
    def _mark_metastases_regions_in_label(self, blank_mask_np, label_np):
        '''Marks the blank mask using the given label image.

        The images are simply labeled with 255 for positive and 0 for negative.

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
        for line in csvreader:
            tif_path, label_tif_path, label_xml_path, release_group, centre = line
            tif_path = Path(tif_path)
            label_tif_path = (label_tif_path or None) and Path(label_tif_path)
            label_xml_path = (label_xml_path or None) and Path(label_xml_path)

            kwargs = {
                'tif_path': tif_path,
                'centre': Centre(int(centre)),
                'label_tif_path': label_tif_path,
                'label_xml_path': label_xml_path,
            }

            if release_group == '16':
                data = C16WSIData(**kwargs)
            elif release_group == '17':
                data = C17WSIData(**kwargs)

            wsi_data.append(data)
    return wsi_data


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


if __name__ == '__main__':
    ds = parse_dataset('/media/shishigami/CVMIGDataDrive/CAMELYON16/window_slide_images/training/C16DataList.csv')
    print(ds)

    sys.exit()
    _TIF_PATH = Path('/media/shishigami/CVMIGDataDrive/CAMELYON16/' +
                     'window_slide_images/training/tumor/Tumor_078.tif')
    _LABEL_PATH = Path('/media/shishigami/CVMIGDataDrive/CAMELYON16' +
                       '/window_slide_images/training/ground_truth/Masks' +
                       '/Tumor_078_Mask.tif')
    # _TIF_PATH = Path('/media/shishigami/CVMIGDataDrive/CAMELYON16/' +
    #                  'window_slide_images/training/normal/Normal_021.tif')
    # _LABEL_PATH = None
    d = C16WSIData(_TIF_PATH, Centre.CENTRE_2, _LABEL_PATH)

    lvl = 5
    # img = d.get_full_wsi_image(lvl)
    # roi_mask = d.get_roi_mask(lvl)
    # metastases_mask = d.get_metastases_mask(lvl)
    # merged_mask = np.full(roi_mask.shape, 0, dtype=np.uint8)
    # merged_mask[np.where(roi_mask)] = 1
    # merged_mask[np.where(metastases_mask)] = 2
    # f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
    # axarr[0].imshow(img)
    # axarr[1].imshow(roi_mask)
    # axarr[1].imshow(metastases_mask)
    # axarr[1].imshow(merged_mask)
    # plt.show()

    patch, label = d.read_region_and_label((41300, 78300), 1, (513 * 5, 513 * 5))
    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
    axarr[0].imshow(patch)
    axarr[1].imshow(label)
    plt.show()
