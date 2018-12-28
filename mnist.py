import gzip

TEST_IMAGES = "data/test_images.gz"
TEST_LABELS = "data/test_labels.gz"
TRAIN_IMAGES = "data/train_images.gz"
TRAIN_LABELS = "data/train_labels.gz"


def _load(images: str, labels: str):
    images = gzip.open(images)
    labels = gzip.open(labels)
    assert int.from_bytes(images.read(4), 'big') == 2051, "invalid magic number"
    assert int.from_bytes(labels.read(4), 'big') == 2049, "invalid magic number"
    size = int.from_bytes(images.read(4), 'big')
    assert int.from_bytes(labels.read(4), 'big') == size, "different size"
    rows = int.from_bytes(images.read(4), 'big')
    cols = int.from_bytes(images.read(4), 'big')
    for i in range(size):
        label = int.from_bytes(labels.read(1), 'big')
        image = [int.from_bytes(images.read(1), 'big') / 255 for _ in range(rows * cols)]
        yield image, label


def load_train():
    return _load(TRAIN_IMAGES, TRAIN_LABELS)


def load_test():
    return _load(TEST_IMAGES, TEST_LABELS)


def print_image(image: list):
    assert len(image) == 28 ** 2, 'invalid image size'
    out = ""
    for i in range(28):
        for j in range(28):
            color = int(image[i * 28 + j] * 255)
            out += f"\033[38;2;{color};{color};{color}m" + "\u2588" * 2
        out += "\n"
    print(out, end='\033[0m')
