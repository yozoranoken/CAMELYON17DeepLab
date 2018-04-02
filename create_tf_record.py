#! /usr/bin/env python3
from argparse import ArgumentParser
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from scipy import io as spio
from scipy.misc import imread
import tensorflow as tf


parser = ArgumentParser(
    description='Create TFRecords from dataset.'
)

parser.add_argument(
    '--data-dir',
    type=Path,
    help='Path to the directory containing the data.',
    required=True,
)

parser.add_argument(
    '--output-path',
    type=Path,
    required=True,
    help='Path to the directory to create TFRecords outputs.',
)

parser.add_argument(
    '--data-list',
    type=Path,
    required=True,
    help='Path to the file listing the data.',
)

parser.add_argument(
    '--image-data-dir',
    type=str,
    default='data',
    help='The directory containing the image data.',
)

parser.add_argument(
    '--label-data-dir',
    type=str,
    default='labels',
    help='The directory containing the augmented label data.',
)

parser.add_argument(
    '--output-dir',
    type=str,
    default='TFRecords',
    help='The directory to contain the TFRecords.',
)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_dataset(
        output_filename,
        images_dir,
        labels_dir,
        filename_list):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i, image_name in enumerate(filename_list):
        if i % 500 == 0:
            tf.logging.info('On image %d of %d', i, len(filename_list))

        image_path = images_dir / (image_name + '.jpeg')
        annotation_path = labels_dir / (image_name + '.mat')


        if not image_path.is_file():
            tf.logging.warning('Could not find {}, ignoring example.'
                               .format(str(image_path)))
            continue
        if not annotation_path.is_file():
            tf.logging.warning('Could not find {}, ignoring example.'
                               .format(str(annotation_path)))
            continue


        image_np = imread(str(image_path))
        annotation_np = spio.loadmat(str(annotation_path))['data']

        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()
        annotation_raw = annotation_np.tostring()

        try:
            example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_h),
                    'width': _int64_feature(image_w),
                    'image_raw': _bytes_feature(img_raw),
                    'annotation_raw': _bytes_feature(annotation_raw)}))
            writer.write(example.SerializeToString())
        except ValueError:
            tf.logging.warning('Invalid example: {}, ignoring.'
                                .format(image_name))

    tf.logging.info('End of TfRecord. Total of image written: {}'.format(i))
    writer.close()

def main(args):
    images_dir = args.data_dir / args.image_data_dir
    labels_dir = args.data_dir / args.label_data_dir
    tf.logging.info('Reading image data from {}'.format(str(images_dir)))
    tf.logging.info('Reading label data from {}'.format(str(labels_dir)))

    with open(str(args.data_list)) as data_list_file:
        images_filename_list = [line.strip() for line in data_list_file]
    tf.logging.info('Image count: {}'.format(len(images_filename_list)))

    np.random.shuffle(images_filename_list)
    split = int(0.10*len(images_filename_list))
    val_images_filename_list = images_filename_list[:split]
    train_images_filename_list = images_filename_list[split:]
    tf.logging.info('Train set size: {}'.format(len(train_images_filename_list)))
    tf.logging.info('Validation set size: {}'.format(len(val_images_filename_list)))

    output_dir = args.output_path / args.output_dir
    output_dir.mkdir(exist_ok=True)

    train_filename = str(output_dir / 'train.tfrecords')
    validation_filename = str(output_dir / 'validation.tfrecords')

    create_tfrecord_dataset(
        train_filename,
        images_dir,
        labels_dir,
        train_images_filename_list,
    )

    create_tfrecord_dataset(
        validation_filename,
        images_dir,
        labels_dir,
        val_images_filename_list,
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(parser.parse_args())
