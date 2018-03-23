#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from xml.etree import ElementTree as ET

from openslide import OpenSlide
from PIL import Image
from PIL import ImageDraw
from utils import str2bool


GROUP_METASTASES = 'metastases'
GROUP_NORMAL = 'normal'

parser = ArgumentParser(
    description=(
        'Utility for converting CAMELYON17 XML lession annotations to ' +
        ' TIF images.'
    ),
)

parser.add_argument(
    '--input', '-i',
    type=Path,
    help='The path to the input XML annotation.',
    required=True,
    metavar='WSI_XML_ANNOTATION',
)

parser.add_argument(
    '--source-image', '-s',
    type=Path,
    help='The path to the TIF source image.',
    required=True,
    metavar='WSI_TIF_IMG',
)

parser.add_argument(
    '--output', '-o',
    type=Path,
    help='Output directory to place PNG annotation.',
    required=True,
    metavar='OUTPUT_DIR',
)

parser.add_argument(
    '--verbose', '-v',
    type=str2bool,
    nargs='?',
    const=True,
    default=True,
    help='Make program verbose.',
)


def read_polygon(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    normal = []
    metastases = []
    annotations = root[0]
    for annotation in annotations:
        polygon = []
        coordinates = annotation[0]
        for coord in coordinates:
            cx = float(coord.attrib['X'])
            cy = float(coord.attrib['Y'])
            polygon.append((cx, cy))

        if annotation.attrib['PartOfGroup'] == GROUP_METASTASES:
            metastases.append(polygon)
        elif annotation.attrib['PartOfGroup'] == GROUP_NORMAL:
            normal.append(polygon)

    return metastases, normal


def create_label(source_image_path, metastases, normal):
    wsi = OpenSlide(str(source_image_path))
    dim = wsi.dimensions

    img_label = Image.new('RGB', dim)
    label_draw = ImageDraw.Draw(img_label, 'RGB')

    for annotation in metastases:
        label_draw.polygon(annotation, fill=(255, 255, 255))

    for annotation in normal:
        label_draw.polygon(annotation, fill=(0, 0, 0))

    del label_draw

    return img_label


def main(args):
    print('[XML2TIF]')

    input_wsi_path = args.input
    output_dir = args.output
    label_filename = input_wsi_path.stem + '.tif'
    output_wsi_path = output_dir / label_filename

    if args.verbose:
        print('Input:', str(input_wsi_path))
        print('Output:', str(output_wsi_path))

    output_dir.mkdir(parents=True, exist_ok=True)

    metastases, normal = read_polygon(input_wsi_path)
    img_label = create_label(args.source_image, metastases, normal)

    img_label.save(str(output_wsi_path), "TIFF")


if __name__ == '__main__':
    main(parser.parse_args())
