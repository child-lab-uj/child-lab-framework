from sys import argv, stdout

import cv2

if __name__ == '__main__':
    _, source, destination, start, end, *_ = argv

    start = float(start)
    end = float(end)

    decoder = cv2.VideoCapture(source)
    width = decoder.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = decoder.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = decoder.get(cv2.CAP_PROP_FPS)

    encoder = cv2.VideoWriter(
        destination,
        fourcc=cv2.VideoWriter.fourcc(*'mp4v'),
        fps=int(fps),
        frameSize=(int(width), int(height)),
        isColor=True,
    )

    skip = int(start * 60 * fps)
    stop = int(end * 60 * fps)

    frame_count = 0

    while True:
        progress = frame_count / stop * 100
        stdout.write(f'\rProgress: {progress:.2f}%')
        stdout.flush()

        success, frame = decoder.read()

        frame_count += 1

        if not success:
            raise RuntimeError(f'Failed to read frame {frame_count}')

        if frame_count <= skip:
            continue

        if frame_count >= stop:
            break

        encoder.write(frame)

    print('\nVideo processed successfully')

    encoder.release()
    decoder.release()
