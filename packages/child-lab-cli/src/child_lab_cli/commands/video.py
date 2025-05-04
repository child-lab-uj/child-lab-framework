import shutil
from collections.abc import Generator
from pathlib import Path
from time import strptime

import click
import torch
from tqdm import trange
from video_io.reader import Reader
from video_io.writer import Writer

from child_lab_cli.workspace.model import Video, Workspace


@click.group('video')
def video() -> None: ...


@video.command('cut')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video-names', type=str, nargs=-1, metavar='<videos>')
@click.option(
    '--time',
    type=str,
    nargs=2,
    required=True,
    help='Start and end of the cut video in format <minutes:seconds>',
)
@click.option(
    '--shape',
    nargs=2,
    type=int,
    required=False,
    default=(1080, 1920),
    help='Common height and width of the result videos',
)
@click.option(
    '--fps',
    type=int,
    required=False,
    default=50,
    help='Common FPS of the result videos',
)
@click.option(
    '--batch-size',
    type=int,
    required=False,
    default=64,
    help='Number of frames to load and save in one step',
)
@click.option(
    '--device',
    type=str,
    required=False,
    default='cpu',
    help='Torch device used for video decoding',
)
def cut(
    workspace_root: Path,
    video_names: list[str],
    time: tuple[str, str],
    shape: tuple[int, int],
    fps: int,
    batch_size: int,
    device: str,
) -> None:
    workspace = Workspace.in_directory(workspace_root)

    available_videos = workspace.videos()
    input_videos: list[Video] = []

    for video_name in video_names:
        video = available_videos.find(lambda video: video.name == video_name)
        if video is None:
            raise FileNotFoundError(f'Video {video_name} not found in {workspace.input}.')

        input_videos.append(video)

    temporary_output = workspace.root / '.cut_tmp'
    if temporary_output.is_dir():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(exist_ok=True)

    start_second = __parse_seconds(time[0])
    end_second = __parse_seconds(time[1])
    frames_to_read = end_second * fps - start_second * fps
    n_batches = frames_to_read // batch_size
    last_batch_size = frames_to_read % batch_size

    video_progress_bar = trange(len(input_videos), desc='Processing videos')
    frame_progress_bar = trange(
        0,
        frames_to_read,
        batch_size,
        desc='Processing',
    )

    for video in input_videos:
        temporary_destination = (temporary_output / video.name).with_suffix(
            video.location.suffix
        )
        destination = (workspace.input / video.name).with_suffix(video.location.suffix)

        reader = Reader(video.location, torch.device(device), fps, shape[1], shape[0])
        writer = Writer(temporary_destination, reader.metadata)

        frame_progress_bar.reset()
        frame_progress_bar.set_description(f'Processing "{video.name}" (batched)')

        def sizes() -> Generator[int, None, None]:
            for _ in range(n_batches):
                yield batch_size
            yield last_batch_size

        for size in sizes():
            batch = reader.read_batch(size)
            assert batch is not None
            writer.write_batch(batch)

            frame_progress_bar.update()

        temporary_destination.rename(destination)

        video_progress_bar.update()

    shutil.rmtree(temporary_output)
    click.echo('Done!')


def __parse_seconds(encoded_time: str) -> int:
    parsed = strptime(encoded_time, '%M:%S')
    return parsed.tm_sec + 60 * parsed.tm_min
