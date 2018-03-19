1. Install openslide on system.
    - Ubuntu: apt-get install openslide-tools
    - MAC: brew install openslide

2. Install anaconda
    - Install packages with `conda install --name camelyon17-deeplab --file req.txt`

3. Put all environment variables in .env

Environment Variables:
- WSI_LIST: Path to the file containing the list of WSI paths and label paths. Each line contains both paths separated by a space. Label paths are optional, since normal slides don't have them.
