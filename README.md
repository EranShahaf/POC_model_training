# CCTV Weapon Threat Data Preparation

This project prepares raw CCTV video datasets (e.g., SCVD, UCF Crime, generic action-recognition sets) into a structure that can be used for:

- **YOLO-based weapon detection** (pistol, rifle, knife)
- **DeepSORT / ByteTrack tracking** and temporal models (\"threat persistence\", weapon drawing)

All of the logic lives in `data_preparation.ipynb`. This README documents the **expected directory layout** and **file formats** that the notebook reads from and writes to.

---

## Top-Level Project Layout

At minimum, the project is expected to contain:

- `data_preparation.ipynb` – main notebook implementing the pipeline.
- `README.md` – this documentation.
- `raw_videos/` – root folder containing *input* CCTV videos, optionally grouped by dataset.
- `prepared_data/` – auto-created output root for processed frames, YOLO labels, and metadata.

You can create additional folders (e.g., `scripts/`, `models/`) as needed; they are not required by the notebook.

```text
POC_model_training/
├── data_preparation.ipynb
├── README.md
├── raw_videos/
│   ├── SCVD/
│   │   ├── scvd_video_001.mp4
│   │   └── ...
│   ├── UCF_Crime/
│   │   ├── robbery_001.mp4
│   │   └── ...
│   ├── ActionDataset/
│   │   ├── action_001.avi
│   │   └── ...
│   └── ... (any other dataset folders or loose video files)
└── prepared_data/
    ├── frames/
    │   └── <video_id>/
    │       ├── <video_id>_frame_000000.jpg
    │       ├── <video_id>_frame_000005.jpg
    │       └── ...
    ├── labels/
    │   └── <video_id>/
    │       ├── <video_id>_frame_000000.txt
    │       ├── <video_id>_frame_000005.txt
    │       └── ...
    └── metadata.csv
```

`prepared_data/` and its subdirectories are created automatically by the notebook when you run it.

---

## Input Structure: `raw_videos/`

The notebook expects the variable:

```python
RAW_VIDEO_ROOT = Path("./raw_videos")
```

You can customize this path in the notebook if your videos are stored elsewhere.

**Allowed formats** (by default): `.mp4`, `.avi`, `.mov`, `.mkv`.

**Example layout:**

```text
raw_videos/
├── SCVD/
│   ├── scvd_cam1_seq01.mp4
│   ├── scvd_cam1_seq02.mp4
│   └── ...
├── UCF_Crime/
│   ├── Assault001.mp4
│   ├── Burglary023.mp4
│   └── ...
├── ActionDataset/
│   ├── fight_001.avi
│   └── ...
└── misc_sources/
    ├── cctv_gun_entrance.mkv
    └── ...
```

The notebook recursively discovers all videos under `RAW_VIDEO_ROOT` using `Path.rglob("*")`. The **video ID** used to group frames is, by default, the file stem:

- video path: `raw_videos/SCVD/scvd_cam1_seq01.mp4`
- video id: `scvd_cam1_seq01`

> If you need more unique IDs (e.g., to encode dataset names), you can modify `get_video_id` in the notebook.

---

## Output Structure: `prepared_data/`

Configured in the notebook as:

```python
OUTPUT_ROOT = Path("./prepared_data")
FRAMES_DIR = OUTPUT_ROOT / "frames"          # frames/<video_id>/*.jpg
LABELS_DIR = OUTPUT_ROOT / "labels"          # labels/<video_id>/*.txt
METADATA_PATH = OUTPUT_ROOT / "metadata.csv"
```

The notebook will automatically create `prepared_data/`, `frames/`, and `labels/` if they do not already exist.

### 1. Frames Directory

For each discovered video:

- A subfolder is created under `prepared_data/frames/` named with the **video ID**.
- Sampled frames are saved as `.jpg` images, with the original or cropped/reshaped resolution depending on configuration.

**Example:**

```text
prepared_data/
└── frames/
    ├── scvd_cam1_seq01/
    │   ├── scvd_cam1_seq01_frame_000000.jpg
    │   ├── scvd_cam1_seq01_frame_000005.jpg
    │   ├── scvd_cam1_seq01_frame_000010.jpg
    │   └── ...
    └── Assault001/
        ├── Assault001_frame_000000.jpg
        └── ...
```

Frame file naming convention:

```text
<video_id>_frame_<frame_index_zero_padded>.jpg
e.g., scvd_cam1_seq01_frame_000035.jpg
```

The `<frame_index>` refers to the original frame index from the video, before sampling (e.g., every 5th frame).

### 2. YOLO Labels Directory

If you provide bounding box annotations and call `write_yolo_labels_for_frame` from the notebook, per-frame YOLO TXT files will be written under:

```text
prepared_data/labels/<video_id>/
```

Example:

```text
prepared_data/
└── labels/
    ├── scvd_cam1_seq01/
    │   ├── scvd_cam1_seq01_frame_000000.txt
    │   ├── scvd_cam1_seq01_frame_000005.txt
    │   └── ...
    └── Assault001/
        ├── Assault001_frame_000000.txt
        └── ...
```

Each label file contains **one line per object** in the YOLO format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** to \[0, 1\] relative to the frame width and height.

**Class IDs used by default:**

- `0` – pistol
- `1` – rifle
- `2` – knife

You can adjust this mapping in `CLASS_NAME_TO_ID` inside the notebook.

### 3. Metadata File: `prepared_data/metadata.csv`

The notebook aggregates frame-level metadata into a single CSV file:

- Location: `prepared_data/metadata.csv`

Each row corresponds to **one extracted frame**, with columns such as:

- `video_id` – stem of the source video filename.
- `frame_idx` – original frame index from the video.
- `frame_filename` – e.g., `scvd_cam1_seq01_frame_000035.jpg`.
- `frame_path` – absolute path to the saved frame.
- `timestamp_sec` – timestamp in seconds, derived from frame index and FPS.
- `width`, `height` – frame resolution after ROI/resize.
- `roi_pixels` – ROI in pixel coordinates `(x_min, y_min, x_max, y_max)` (or `NaN`/`None` if no ROI used).
- `roi_normalized` – ROI normalized to \[0, 1\] relative to the frame.
- `is_active_threat` – boolean flag based on temporal action annotations.
- `action_label` – semantic tag (e.g., `Active_Threat`, `Weapon_Visible`, etc.).
- `yolo_label_path` – absolute path to the YOLO TXT file for this frame, if it exists.

This CSV is intended for ingestion by a **Database Module** or custom dataloaders for training/analysis.

---

## Action Sequence Annotations (Optional)

To tag sequences where a weapon is being drawn or an active threat is present, the notebook uses an `ACTION_SEQUENCES` dictionary:

```python
ACTION_SEQUENCES = {
    "scvd_cam1_seq01": [
        {"start_sec": 12.0, "end_sec": 25.0, "label": "Active_Threat"},
        {"start_sec": 40.0, "end_sec": 55.0, "label": "Weapon_Visible"},
    ],
    "Assault001": [
        {"start_sec": 5.0, "end_sec": 18.0, "label": "Active_Threat"},
    ],
}
```

You can:

- Hard-code this dict directly in `data_preparation.ipynb`, **or**
- Load it from an external CSV/JSON file that you place anywhere in the project (e.g., `annotations/action_sequences.json`) and parse it inside the notebook.

These annotations are used during frame extraction to populate `is_active_threat` and `action_label` in `metadata.csv`.

---

## Running the Notebook End-to-End

1. **Place input videos** under `raw_videos/` (or change `RAW_VIDEO_ROOT` in the notebook to point to your location).
2. (Optional) **Define action sequences** in `ACTION_SEQUENCES` to tag active threat periods.
3. (Optional) **Prepare bounding box annotations** and wire them into `write_yolo_labels_for_frame` for each frame with objects.
4. Open `data_preparation.ipynb` and run cells from top to bottom:
   - Environment setup
   - Configuration
   - Frame extraction
   - Label creation
   - Metadata logging
5. Use the resulting `prepared_data/frames/`, `prepared_data/labels/`, and `prepared_data/metadata.csv` to train YOLO and integrate with DeepSORT/ByteTrack.

This structure keeps your **temporal grouping by video**, per-frame **YOLO labels**, and **sequence-level threat metadata** aligned for robust CCTV weapon detection and tracking.

