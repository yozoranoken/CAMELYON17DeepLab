import csv
from enum import IntEnum
import logging
from pathlib import Path
from random import randint
from xml.etree import ElementTree as ET

import numpy as np
import openslide
from PIL import Image
from PIL import ImageDraw
import pyclipper
from skimage import morphology
from skimage import transform
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


_DOWNSAMPLING_LEVEL = 5
class WSIData:
    _GROUP_METASTASES = ('metastases', '_0', '_1')
    _GROUP_NORMAL = ('normal', '_2')
    _PATCH_SIDE = 513
    PATCH_DIM = (_PATCH_SIDE, _PATCH_SIDE)
    _SMALL_OBJECT_AREA = 512
    _SMALL_HOLE_AREA = 256
    NORMAL_COLOR = (24, 16, 94)
    TUMOR_COLOR = (206, 28, 105)

    class AnnotationLabel(IntEnum):
        NORMAL = 1
        TUMOR = 2

    @staticmethod
    def get_default_downsampling_level():
        return _DOWNSAMPLING_LEVEL


    def __init__(self, wsi_name, tif_path, label_path, release_group, centre):
        self._name = wsi_name
        self._tif_path = tif_path
        self._slide = openslide.OpenSlide(str(self._tif_path))
        self._label_path = label_path or None
        if self._label_path is not None:
            if not self._label_path.is_file():
                raise IOError('File does not exist.')
            self._label = openslide.OpenSlide(str(self._label_path))
        self._centre = centre
        self._release_group = release_group
        self._roi = None
        self._metastases_mask = None
        self._normal_mask = None
        # self._read_polygon()

    # def _read_polygon(self):
    #     normal = []
    #     metastases = []

    #     if self._label_path is not None:
    #         tree = ET.parse(str(self._label_path))
    #         root = tree.getroot()

    #         annotations = root[0]
    #         for annotation in annotations:
    #             polygon = []
    #             coordinates = annotation[0]
    #             for coord in coordinates:
    #                 cx = round(float(coord.attrib['X']))
    #                 cy = round(float(coord.attrib['Y']))
    #                 polygon.append((cx, cy))

    #             if (annotation.attrib['PartOfGroup'] in
    #                     self._GROUP_METASTASES):
    #                 metastases.append(polygon)
    #             elif (annotation.attrib['PartOfGroup'] in
    #                     self._GROUP_NORMAL):
    #                 normal.append(polygon)

    #     self._metastases_label = metastases
    #     self._normal_label = normal

    @property
    def name(self):
        return self._name

    @property
    def tif_path(self):
        return self._tif_path

    @property
    def label_path(self):
        return self._label_path

    @property
    def centre(self):
        return self._centre

    def get_dimensions(self, ds_level=_DOWNSAMPLING_LEVEL):
        return self._slide.level_dimensions[ds_level]

    def get_image(self, level_downsample=_DOWNSAMPLING_LEVEL):
        return self._slide.read_region(
            location=(0, 0),
            level=level_downsample,
            size=self._slide.level_dimensions[level_downsample],
        ).convert('RGB')

    def _downsample_label_coords(self, labels, level):
        if not labels:
            return []

        ds = self._slide.level_downsamples[level]
        round10s = lambda t: (round(t[0] / ds), round(t[1] / ds))
        return [[round10s(cell) for cell in r]
                for r in labels]

    # def get_metastases_label(self, level=0):
    #     return self._downsample_label_coords(self._metastases_label,
    #                                          level)

    # def get_normal_label(self, level=0):
    #     return self._downsample_label_coords(self._normal_label,
    #                                          level)

    # @staticmethod
    # def _coords2mask(dim, pos, neg=None):
    #     mask = Image.new('RGB', dim)
    #     label_draw = ImageDraw.Draw(mask, 'RGB')


    #     for annotation in pos:
    #         label_draw.polygon(annotation, fill='white')

    #     if neg:
    #         for annotation in neg:
    #             label_draw.polygon(annotation, fill=(0, 0, 0))

    #     return mask

    def get_metastases_mask(self, level=_DOWNSAMPLING_LEVEL):
        if self._metastases_mask is None:
            sw, sh = self._slide.level_dimensions[level]
            label_np = np.full((sh, sw), False)
            if self._label_path is not None:
                img = self._read_slide_image(self._label, level)
                img_np = np.array(img.convert('L'))

                if self._release_group == 16:
                    lh, lw = img_np.shape
                    img_np = img_np[:min(sh, lh), :min(sw, lw)]
                    label_np[np.where(img_np > 0)] = True

                    # from matplotlib import pyplot as plt

                    # plt.imshow(label_np)
                    # plt.show()

                    # import sys
                    # sys.exit()
            self._metastases_mask = label_np

        return self._metastases_mask

    def get_normal_mask(self, level=_DOWNSAMPLING_LEVEL):
        '''Returns a 2D Boolean matrix for the normal ROI of the WSI.
        '''
        if self._normal_mask is None:
            roi = self._load_roi()
            metastases = self.get_metastases_mask(level)
            self._normal_mask = np.bitwise_xor(roi, metastases)

        return self._normal_mask

    def _make_coordinates_relative(self, src_pt, regions):
        return [[(x - src_pt[0], y - src_pt[1]) for x, y in region]
                for region in regions]

    def _clip_patch(self, subject, patch_coord, patch_dim):
        w, h = patch_dim
        clipper = (
            (0, 0),
            (w, 0),
            (w, h),
            (0, h),
        )

        subject_relative = self._make_coordinates_relative(
            patch_coord,
            subject,
        )

        pc = pyclipper.Pyclipper()
        pc.AddPath(clipper, pyclipper.PT_CLIP, True)
        pc.AddPaths(subject_relative, pyclipper.PT_SUBJECT, True)

        solution = pc.Execute(
            pyclipper.CT_INTERSECTION,
            pyclipper.PFT_EVENODD,
            pyclipper.PFT_EVENODD,
        )

        solution = [[tuple(c) for c in s] for s in solution]
        return solution

    def _read_tumor_patch(self, coords):
        patch = (self._label.read_region(coords, 0, self.PATCH_DIM)
                 .convert('L'))
        label_patch = np.array(patch, dtype=np.uint8)

        if self._release_group == 16:
            tumor_coords = np.where(label_patch > 0)
        elif self._release_group == 17:
            tumor_coords = np.where(label_patch == 1)

        label_patch = np.full(label_patch.shape, False)
        label_patch[tumor_coords] = True

        # from matplotlib import pyplot as plt
        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        # ax1.imshow(patch)
        # ax2.imshow(label_patch)
        # plt.show()
        # import sys
        # sys.exit()


        return label_patch

    def read_region(self, ds_coord, ds_level=_DOWNSAMPLING_LEVEL):

        round_ds_factor = 2**ds_level
        w, h = self._slide.dimensions
        w_ds, h_ds = self._slide.level_dimensions[ds_level]
        w_f = w / w_ds
        h_f = h / h_ds
        pad = self._PATCH_SIDE // 2
        cx_ds, cy_ds = ds_coord
        cx = round(cx_ds * round_ds_factor) + pad * randint(-1, 1) + randint(0, round_ds_factor)
        cy = round(cy_ds * round_ds_factor) + pad * randint(-1, 1) + randint(0, round_ds_factor)
        coords = max(pad, min(cx, w - pad)), max(pad, min(cy, h - pad))

        # import sys
        # sys.exit()

        # img = self._read_slide_image(self._slide)
        # img_np = np.array(img)
        # tumor_mask = self.get_metastases_mask()
        # r = img_np[:, :, 0]
        # g = img_np[:, :, 1]
        # b = img_np[:, :, 2]
        # inv = np.invert(tumor_mask)
        # r[inv] = 0
        # g[inv] = 0
        # b[inv] = 0
        # img_np[:, :, 0] = r
        # img_np[:, :, 1] = g
        # img_np[:, :, 2] = b

        # from matplotlib import pyplot as plt
        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        # ax1.imshow(img)
        # ax2.imshow(img_np)
        # plt.show()
        # import sys
        # sys.exit()



        region = self._slide.read_region(
            coords, 0, self.PATCH_DIM).convert('RGB')
        # self._load_roi()  #! don't reposition line
        # channel_h, channel_s = self._extract_hs_channels(region)

        # # Patch; 2D bool array; True on lymph node ROI
        # region_roi = self._merge_hs_channels_to_mask(
        #     channel_h,
        #     channel_s,
        # )
        # kernel = morphology.disk(3)
        # region_roi = morphology.binary_opening(
        #     region_roi, kernel)

        label_np = np.full(self.PATCH_DIM, int(self.AnnotationLabel.NORMAL),
                           dtype=np.uint8)
        # label_np[region_roi] = int(self.AnnotationLabel.NORMAL)

        label_img = np.zeros(label_np.shape + (3,), dtype=np.uint8)
        # color_image(label_img, region_roi, self.NORMAL_COLOR)

        if self._label_path is not None:
            # Patch; 2D bool array; True on tumor region
            tumor_roi = self._read_tumor_patch(coords)
            label_np[tumor_roi] = int(self.AnnotationLabel.TUMOR)
            color_image(label_img, tumor_roi, self.TUMOR_COLOR)


        # from matplotlib import pyplot as plt

        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        # ax1.imshow(region)
        # ax2.imshow(label_img)
        # plt.show()

        # import sys
        # sys.exit()

        # patch_meta = []
        # patch_norm = []
        # if self._metastases_label:
        #     patch_meta = self._clip_patch(
        #         self._metastases_label,
        #         (cx, cy),
        #         self.PATCH_DIM,
        #     )

        #     if self._normal_label:
        #         patch_norm = self._clip_patch(
        #             self._normal_label,
        #             (cx, cy),
        #             self.PATCH_DIM,
        #         )
        # label = self._coords2mask(self.PATCH_DIM, patch_meta, patch_norm)
        # return region, label

        ## PIL, PIL, np.arr
        return region, Image.fromarray(label_img), label_np

    @staticmethod
    def _extract_hs_channels(img):
        img_rgb = np.array(img, dtype=np.uint8)
        img_hsv = rgb2hsv(img_rgb)
        channel_h = img_hsv[:, :, 0]
        channel_s = img_hsv[:, :, 1]
        return channel_h, channel_s

    def _merge_hs_channels_to_mask(self,
                                   channel_h,
                                   channel_s,
                                   small_object_area=None,
                                   small_hole_area=None):
        binary_h = channel_h > self._thresh_h
        binary_s = channel_s > self._thresh_s
        binary = np.bitwise_and(binary_h, binary_s)

        binary = morphology.remove_small_objects(
            binary, small_object_area or self._SMALL_OBJECT_AREA)
        binary = morphology.remove_small_holes(
            binary, small_hole_area or self._SMALL_HOLE_AREA)
        return binary

    @staticmethod
    def _read_slide_image(slide, level=_DOWNSAMPLING_LEVEL):
        return slide.read_region(
            location=(0, 0),
            level=_DOWNSAMPLING_LEVEL,
            size=slide.level_dimensions[_DOWNSAMPLING_LEVEL],
        ).convert('RGB')

    def _load_roi(self):
        if self._roi is None:
            slide_img = self._read_slide_image(self._slide)

            channel_h, channel_s = self._extract_hs_channels(slide_img)

            self._thresh_h = threshold_otsu(channel_h)
            self._thresh_s = threshold_otsu(channel_s)

            self._roi = self._merge_hs_channels_to_mask(channel_h, channel_s)
        return self._roi


def read_wsi_list_file(filelist_path):
    wsi_data = []
    with open(str(filelist_path)) as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        for line in csvreader:
            tif_path, label_path, release_group, centre = line
            tif_path = Path(tif_path)
            label_path = (label_path or None) and Path(label_path)
            data = WSIData(wsi_name=tif_path.stem,
                           tif_path=tif_path,
                           label_path=label_path,
                           release_group=int(release_group),
                           centre=int(centre))
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

def color_image(img_np, mask, color):
    for i, c in enumerate(color):
        channel = img_np[:,:,i]
        channel[mask] = c
        img_np[:,:,i] = channel
