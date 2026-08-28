import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoFrameDataset(Dataset):

    def __init__(
        self,
        video_path,
        frame_interval=5,
        image_size=128
    ):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

        self.samples = []

        self._build_index()

    def _build_index(self):

        cap = cv2.VideoCapture(
            self.video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{self.video_path}"
            )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()

        for frame_idx in range(
            0,
            frame_count,
            self.frame_interval
        ):
            self.samples.append(frame_idx)

        print(
            f"Video frames: {frame_count}"
        )

        print(
            f"Training samples: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        frame_idx = self.samples[index]

        cap = cv2.VideoCapture(
            self.video_path
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        success, frame = cap.read()

        cap.release()

        if not success:
            raise RuntimeError(
                f"Could not read frame "
                f"{frame_idx}"
            )

        # BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize + [0,1]
        frame = self.transform(frame)

        return frame