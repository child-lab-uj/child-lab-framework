from dataclasses import dataclass
from pprint import pprint
import yaml


@dataclass
class Video:
    camera_label: str
    filepath: str


@dataclass
class Input:
    preset_label: str
    calibration_videos: list[Video]
    transformation_videos: list[Video]


def parse_input_file(file_path: str) -> Input:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

        preset_label = data['preset_label']

        calibration_videos = [Video(**video) for video in data['calibration_videos']]
        transformation_videos = [
            Video(**video) for video in data['transformation_videos']
        ]

        return Input(preset_label, calibration_videos, transformation_videos)


if __name__ == '__main__':
    config = parse_input_file('input_example.yaml')
    pprint(config)
