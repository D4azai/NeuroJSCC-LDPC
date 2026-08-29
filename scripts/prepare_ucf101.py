from pathlib import Path
import shutil
import random


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this ONLY if your UCF101 folder has another location.
UCF_ROOT = PROJECT_ROOT / "data" / "UCF-101"

# Where the prepared dataset will be created
OUTPUT_ROOT = PROJECT_ROOT / "data"

# Official UCF101 split
SPLIT_NUMBER = 1

# Percentage of official training videos used for validation
VAL_RATIO = 0.10

# Reproducible validation split
RANDOM_SEED = 42

# ------------------------------------------------------------
# Start with these 10 classes.
# We can expand to all 101 later.
# ------------------------------------------------------------

SELECTED_CLASSES = [
    "ApplyEyeMakeup",
    "Archery",
    "BabyCrawling",
    "BasketballShooting",
    "Biking",
    "GolfSwing",
    "PlayingGuitar",
    "PlayingPiano",
    "Surfing",
    "WalkingWithADog",
]


# ============================================================
# FIND SPLIT FILE
# ============================================================

def find_split_file(filename):

    candidates = list(
        PROJECT_ROOT.rglob(filename)
    )

    if not candidates:

        raise FileNotFoundError(
            f"\nCould not find {filename}\n\n"
            "Make sure the UCF101 split files are "
            "inside the project or data directory.\n"
        )

    # Prefer a path containing ucfTrainTestlist
    for candidate in candidates:

        if "ucfTrainTestlist" in str(candidate):
            return candidate

    return candidates[0]


# ============================================================
# READ SPLIT FILE
# ============================================================

def read_train_list(path):

    videos = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # trainlist01.txt usually contains:
            #
            # ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1
            #
            # We only need the path.

            video_path = line.split()[0]

            videos.append(video_path)

    return videos


def read_test_list(path):

    videos = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # testlist01.txt usually contains:
            #
            # ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03.avi

            videos.append(line)

    return videos


# ============================================================
# FILTER SELECTED CLASSES
# ============================================================

def filter_classes(video_list):

    selected = []

    selected_set = set(
        SELECTED_CLASSES
    )

    for relative_path in video_list:

        # Convert Windows/Linux separators
        normalized = relative_path.replace(
            "\\",
            "/"
        )

        class_name = normalized.split("/")[0]

        if class_name in selected_set:
            selected.append(
                normalized
            )

    return selected


# ============================================================
# FIND VIDEO
# ============================================================

def locate_video(relative_path):

    relative_path = Path(
        relative_path
    )

    # Normal expected location:
    candidate = (
        UCF_ROOT
        / relative_path
    )

    if candidate.exists():
        return candidate

    # Fallback: search by filename
    filename = relative_path.name

    matches = list(
        UCF_ROOT.rglob(filename)
    )

    if matches:

        return matches[0]

    return None


# ============================================================
# COPY VIDEO
# ============================================================

def copy_video(
    relative_path,
    split_name
):

    source = locate_video(
        relative_path
    )

    if source is None:

        print(
            f"WARNING: video not found: "
            f"{relative_path}"
        )

        return False

    destination = (
        OUTPUT_ROOT
        / split_name
        / Path(relative_path)
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not destination.exists():

        shutil.copy2(
            source,
            destination
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UCF101 DATASET PREPARATION")
    print("=" * 70)

    print(
        f"UCF101 root:\n{UCF_ROOT}"
    )

    print(
        f"Output root:\n{OUTPUT_ROOT}"
    )

    print()

    # --------------------------------------------------------
    # Check UCF101
    # --------------------------------------------------------

    if not UCF_ROOT.exists():

        raise FileNotFoundError(
            f"\nUCF101 directory does not exist:\n"
            f"{UCF_ROOT}\n\n"
            "If your UCF101 folder has another name/location, "
            "change UCF_ROOT at the top of this script.\n"
        )

    # --------------------------------------------------------
    # Find official split files
    # --------------------------------------------------------

    train_filename = (
        f"trainlist0{SPLIT_NUMBER}.txt"
    )

    test_filename = (
        f"testlist0{SPLIT_NUMBER}.txt"
    )

    train_list_path = find_split_file(
        train_filename
    )

    test_list_path = find_split_file(
        test_filename
    )

    print(
        f"Train split:\n{train_list_path}"
    )

    print(
        f"Test split:\n{test_list_path}"
    )

    print()

    # --------------------------------------------------------
    # Read official lists
    # --------------------------------------------------------

    official_train = read_train_list(
        train_list_path
    )

    official_test = read_test_list(
        test_list_path
    )

    print(
        f"Official training videos: "
        f"{len(official_train)}"
    )

    print(
        f"Official test videos: "
        f"{len(official_test)}"
    )

    # --------------------------------------------------------
    # Keep only selected classes
    # --------------------------------------------------------

    train_videos = filter_classes(
        official_train
    )

    test_videos = filter_classes(
        official_test
    )

    print()
    print(
        f"Selected training videos: "
        f"{len(train_videos)}"
    )

    print(
        f"Selected test videos: "
        f"{len(test_videos)}"
    )

    # --------------------------------------------------------
    # Make validation split from TRAIN only
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    videos_by_class = {}

    for video in train_videos:

        class_name = video.split("/")[0]

        videos_by_class.setdefault(
            class_name,
            []
        ).append(video)

    final_train = []
    val_videos = []

    for class_name, videos in sorted(
        videos_by_class.items()
    ):

        videos = list(videos)

        random.shuffle(
            videos
        )

        val_count = max(
            1,
            int(
                len(videos)
                * VAL_RATIO
            )
        )

        class_val = videos[
            :val_count
        ]

        class_train = videos[
            val_count:
        ]

        val_videos.extend(
            class_val
        )

        final_train.extend(
            class_train
        )

        class_test_count = sum(
            1
            for video_path in test_videos
            if video_path.startswith(class_name + "/")
        )

        print(
            f"{class_name:25s} "
            f"train={len(class_train):4d} "
            f"val={len(class_val):3d} "
            f"test={class_test_count:4d}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        (
            OUTPUT_ROOT / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy videos
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPYING TRAIN VIDEOS")
    print("=" * 70)

    train_ok = 0

    for video in final_train:

        if copy_video(
            video,
            "train"
        ):

            train_ok += 1

    print(
        f"Copied {train_ok}/{len(final_train)} "
        "training videos."
    )

    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPYING VALIDATION VIDEOS")
    print("=" * 70)

    val_ok = 0

    for video in val_videos:

        if copy_video(
            video,
            "val"
        ):

            val_ok += 1

    print(
        f"Copied {val_ok}/{len(val_videos)} "
        "validation videos."
    )

    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COPYING TEST VIDEOS")
    print("=" * 70)

    test_ok = 0

    for video in test_videos:

        if copy_video(
            video,
            "test"
        ):

            test_ok += 1

    print(
        f"Copied {test_ok}/{len(test_videos)} "
        "test videos."
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"TRAIN: {train_ok} videos"
    )

    print(
        f"VAL:   {val_ok} videos"
    )

    print(
        f"TEST:  {test_ok} videos"
    )

    print()

    print(
        "Dataset locations:"
    )

    print(
        OUTPUT_ROOT / "train"
    )

    print(
        OUTPUT_ROOT / "val"
    )

    print(
        OUTPUT_ROOT / "test"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()