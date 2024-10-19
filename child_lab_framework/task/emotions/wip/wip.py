import cv2
from deepface import DeepFace
from retinaface import RetinaFace


def detect_and_crop_faces_with_retinaface(frame):
    """Detects faces in a frame using RetinaFace and returns cropped face regions."""
    frame_height, frame_width, _ = frame.shape
    faces = RetinaFace.detect_faces(frame)
    cropped_faces = []

    for key, face_info in faces.items():
        # Get bounding box values
        identity = face_info['facial_area']
        x1, y1, x2, y2 = identity[0], identity[1], identity[2], identity[3]
        w, h = x2 - x1, y2 - y1

        # Crop face region, adding padding of 40 pixels
        cropped_face = frame[
            max(y1 - 40, 0) : min(y1 + h + 40, frame_height),
            max(x1 - 40, 0) : min(x1 + w + 40, frame_width),
        ]
        cropped_faces.append((cropped_face, x1, y1, w, h))

    return cropped_faces


def score_emotions(emotions):
    # Most of the time, "angry" and "fear" are similar to "neutral" in the reality
    scores = {
        'angry': -0.05,
        'disgust': 0,
        'fear': -0.07,
        'happy': 1,
        'sad': -1,
        'surprise': 0,
        'neutral': 0,
    }
    val = 0
    for emotion, score in scores.items():
        val += emotions['emotion'][emotion] * score

    return val


def process_video(input_video_path, output_video_path, emotion_model='VGG-Face'):
    # Open the video files
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f'Error opening video {input_video_path}')
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter object to save annotated video
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (frame_width, frame_height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and crop faces using RetinaFace
        cropped_faces = detect_and_crop_faces_with_retinaface(frame)

        for cropped_face, x, y, w, h in cropped_faces:
            try:
                # Analyze emotion on cropped face
                analysis = DeepFace.analyze(
                    cropped_face, actions=['emotion'], enforce_detection=False
                )
                emotion = score_emotions(analysis[0])
                print('emotion score: ', emotion, analysis[0])

                # Annotate video with emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                cv2.putText(
                    frame,
                    str(emotion),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

            except Exception as e:
                print(f'Exception in detection: {str(e)}')

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f'Processing completed for {input_video_path} and saved to {output_video_path}')


def main():
    # left_video_path = "dev/data/short/window_left.mp4"
    left_video_path = 'dev/data/short/very_short.avi'
    output_left_video_path = 'output_left_video.avi'

    # Process left video
    process_video(left_video_path, output_left_video_path)


if __name__ == '__main__':
    main()
