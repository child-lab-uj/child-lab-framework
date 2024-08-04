from sys import argv, stdout
import cv2


if __name__ == '__main__':
    _, source, destination, frame, *_ = argv
    frame = int(frame)

    decoder = cv2.VideoCapture(source)

    for i in range(1, frame):
        success, _ = decoder.read()
        assert success
        stdout.write(f'\rRead {i} frames')

    cv2.imwrite(destination, decoder.read()[1])

    print('\nDone!')
