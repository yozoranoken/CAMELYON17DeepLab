#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


import multiresolutionimageinterface as mir


_LABEL_MAP = {
    'metastases': 1,
    '_0': 1,
    '_1': 1,
    'normal': 2,
    '_2': 2,
    'ignore': 3,
}


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--xml-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--wsi-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--names',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--folder-name',
        default='masks',
    )

    return parser.parse_args()


def main(args):
    output_folder = args.output_dir / args.folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    names = []
    with open(str(args.names)) as names_file:
        for name in names_file.readlines():
            names.append(name.strip())

    for xml_path in args.xml_dir.glob('*.xml'):
        stem = xml_path.stem

        if stem not in names:
            continue

        print(f'>> Creating mask for {stem}...')
        reader = mir.MultiResolutionImageReader()

        wsi_tif_path = args.wsi_dir / f'{stem}.tif'
        mr_image = reader.open(str(wsi_tif_path))

        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)

        xml_repository.setSource(str(xml_path))

        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        output_path = output_folder / f'{stem}_Mask.tif'
        annotation_mask.convert(
            annotation_list,
            str(output_path),
            mr_image.getDimensions(),
            mr_image.getSpacing(),
            _LABEL_MAP,
        )
        print(f'   Mask saved for {stem} at {output_path}')


if __name__ == '__main__':
    main(collect_arguments())
