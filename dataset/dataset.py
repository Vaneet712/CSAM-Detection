import os

import torch
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
from tqdm import tqdm

import pillow_avif

from transformers import AutoImageProcessor


# ============================================================
# SIGLIP
# ============================================================

SIGLIP_MODEL = "google/siglip-base-patch16-384"

_processor = None


def get_image_processor():

    global _processor

    if _processor is None:

        _processor = AutoImageProcessor.from_pretrained(
            SIGLIP_MODEL,
            use_fast=False
        )

    return _processor


# ============================================================
# LOGIC VECTOR
# ============================================================

def compute_logic_vector_for_image(
    image_path,
    text
):

    """
    Computes the 5-dimensional logic vector.

    Output:
    [
        nudity_score,
        p_minor_max,
        csam_init,
        adult_abuse,
        bully_score
    ]
    """

    from utils.bully_score import compute_bully_score
    from utils.rule_engine import evaluate_logic

    from models.face_detector import detect_faces
    from models.age_detector import predict_age
    from models.emotion_detector import predict_emotion
    from models.nudity_detector import predict_nudity


    # --------------------------------------------------------
    # Text based bully score
    # --------------------------------------------------------

    bully_score = compute_bully_score(
        text
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception:

        return torch.zeros(
            5,
            dtype=torch.float32
        )


    try:
        nudity_score = predict_nudity(image)
        face_crops, _ = detect_faces(image)

        age_vectors = []
        emotion_vectors = []

        for face_crop in face_crops:
            age_vec = predict_age(face_crop)
            emo_vec = predict_emotion(face_crop)
            age_vectors.append(age_vec)
            emotion_vectors.append(emo_vec)

        if len(age_vectors) == 0:
            age_vectors = [[0.0, 0.0, 1.0]]
            emotion_vectors = [[0.0, 0.0, 0.0, 0.0, 1.0]]

        logic_vec = evaluate_logic(
            age_vectors=age_vectors,
            emotion_vectors=emotion_vectors,
            nudity_score=nudity_score,
            bully_score=bully_score
        )
        return logic_vec
    except Exception:
        return torch.zeros(5, dtype=torch.float32)


# ============================================================
# DATASET
# ============================================================

class CSAMDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        precompute_logic=True,
        skip_caption_loading=False
    ):

        """
        Parameters
        ----------
        csv_file : str
            Metadata CSV.

        image_dir : str
            Root directory containing train/val/test images.

        precompute_logic : bool
            Whether to compute logic vectors beforehand.

        skip_caption_loading : bool
            If True, caption loading from the original
            train/val/test directories is skipped.

            This is required for fixed 5-fold CV splits,
            because captions are already merged into the
            prepared fold CSV files.
        """

        self.csv_file = csv_file

        self.image_dir = image_dir

        self.skip_caption_loading = (
            skip_caption_loading
        )


        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        self.data = pd.read_csv(
            csv_file
        )


        # ----------------------------------------------------
        # Image processor
        # ----------------------------------------------------

        self.processor = get_image_processor()


        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        col_map = {}

        for col in self.data.columns:

            clow = (
                col
                .lower()
                .strip()
            )

            if clow in [
                "image",
                "image_path",
                "filename",
                "file_name",
                "img",
                "image name"
            ]:

                col_map[col] = "image"

            elif clow in [
                "text",
                "description",
                "caption",
                "text_description"
            ]:

                col_map[col] = "text"

            elif clow in [
                "label",
                "target",
                "class",
                "is_csam"
            ]:

                col_map[col] = "label"

            elif clow == "filepath":

                col_map[col] = "filepath"

            elif clow == "subtype":

                col_map[col] = "subtype"

            elif clow == "group_id":

                col_map[col] = "group_id"

            elif clow == "fold":

                col_map[col] = "fold"


        self.data.rename(
            columns=col_map,
            inplace=True
        )


        # ----------------------------------------------------
        # Ensure image column
        # ----------------------------------------------------

        if "image" not in self.data.columns:

            raise ValueError(
                "Metadata CSV does not contain "
                "an image column."
            )


        # ----------------------------------------------------
        # Ensure label
        # ----------------------------------------------------

        if "label" not in self.data.columns:

            raise ValueError(
                "Metadata CSV does not contain "
                "a label column."
            )


        # ----------------------------------------------------
        # Normalize labels
        # ----------------------------------------------------

        self.data["label"] = (

            self.data["label"]

            .astype(str)

            .str.strip()

            .str.upper()
        )


        label_map = {

            "NON_CSAM": 0,

            "CSAM": 1

        }


        invalid_labels = (

            ~self.data["label"].isin(
                label_map.keys()
            )
        )


        if invalid_labels.any():

            bad_labels = (

                self.data.loc[
                    invalid_labels,
                    "label"
                ]

                .unique()

                .tolist()
            )

            raise ValueError(
                f"Invalid labels found: "
                f"{bad_labels}"
            )


        self.data["label"] = (

            self.data["label"]

            .map(label_map)

            .astype(int)
        )


        # ====================================================
        # CAPTION LOADING
        # ====================================================

        if not self.skip_caption_loading:

            self._load_and_merge_captions()

        else:

            # ------------------------------------------------
            # Fixed CV splits already contain "text"
            # ------------------------------------------------

            if "text" not in self.data.columns:

                raise ValueError(
                    "skip_caption_loading=True but "
                    "the CSV does not contain a 'text' column."
                )

            self.data["text"] = (

                self.data["text"]

                .fillna("")

                .astype(str)
            )

            print(
                "[INFO] Caption loading skipped."
            )

            print(
                "[INFO] Using captions already present "
                "in the CSV."
            )


        # ====================================================
        # LOGIC VECTOR COLUMNS
        # ====================================================

        self.has_logic_cols = all(

            col in self.data.columns

            for col in [

                "nudity_score",
                "p_minor_max",
                "csam_init",
                "adult_abuse",
                "bully_score"

            ]

        )


        # ====================================================
        # LOGIC PRECOMPUTATION
        # ====================================================

        self.precompute_logic = (
            precompute_logic
        )

        self.logic_cache = {}


        if (

            self.precompute_logic

            and not self.has_logic_cols

        ):

            print(
                f"Pre-computing logic vectors "
                f"for {len(self.data)} dataset items..."
            )


            for idx in tqdm(
                range(len(self.data)),
                desc="Pre-computing logic vectors",
                leave=False
            ):

                row = self.data.iloc[
                    idx
                ]


                img_path = (

                    self._resolve_image_path(
                        row["image"]
                    )

                )


                text = row["text"]


                lvec = (

                    compute_logic_vector_for_image(
                        img_path,
                        text
                    )

                )


                self.logic_cache[idx] = lvec


# ============================================================
# CAPTION LOADER
# ============================================================

    def _load_and_merge_captions(
        self
    ):

        """
        Automatically loads captions from the original
        dataset_updated_organized directory.

        This function is used for the original
        train/val/test metadata CSVs.

        For fixed 5-fold CSVs use:

            skip_caption_loading=True
        """


        # ====================================================
        # DETERMINE SPLIT
        # ====================================================

        normalized_path = (

            os.path.abspath(
                self.csv_file
            )

            .replace(
                "\\",
                "/"
            )

            .lower()

        )


        if "/train/" in normalized_path or "train" in normalized_path:
            split = "train"
        elif "/val/" in normalized_path or "val" in normalized_path:
            split = "val"
        elif "/test/" in normalized_path or "test" in normalized_path:
            split = "test"
        else:
            split = "all"

        # ====================================================
        # DATASET ROOT
        # ====================================================
        metadata_dir = os.path.dirname(os.path.abspath(self.csv_file))
        dataset_root = os.path.dirname(metadata_dir)

        print()
        print("=" * 70)
        print(f"CAPTION LOADING: {split.upper()}")
        print("=" * 70)
        print(f"[INFO] Dataset root:\n{dataset_root}")

        # ====================================================
        # CAPTION FILE PATHS
        # ====================================================
        caption_files = []
        possible_caption_paths = [
            os.path.join(self.image_dir, "csam_non_sens.csv"),
            os.path.join(self.image_dir, "non_csam.csv"),
            os.path.join(dataset_root, "csam_non_sens.csv"),
            os.path.join(dataset_root, "non_csam.csv"),
            os.path.join(dataset_root, split, f"{split}_csam_non_sens.csv"),
            os.path.join(dataset_root, split, f"{split}_non_csam.csv"),
        ]
        for cap_path in possible_caption_paths:
            if os.path.exists(cap_path) and cap_path not in caption_files:
                caption_files.append(cap_path)

        if len(caption_files) == 0:
            raise FileNotFoundError(
                f"No caption CSV files found for {split}.\n\n"
                f"Checked:\n"
                + "\n".join(possible_caption_paths)
            )


        print(
            "[INFO] Caption files:"
        )


        for caption_file in caption_files:

            print(
                f"       {caption_file}"
            )


        # ====================================================
        # READ CAPTION FILES
        # ====================================================

        caption_frames = []


        for caption_file in caption_files:

            df = pd.read_csv(
                caption_file
            )


            print(

                f"[INFO] "
                f"{os.path.basename(caption_file)} "
                f"rows = {len(df)}"

            )


            required_columns = {

                "filename",
                "caption"

            }


            missing_columns = (

                required_columns
                -
                set(df.columns)

            )


            if missing_columns:

                raise ValueError(

                    f"Caption file:\n"
                    f"{caption_file}\n"

                    f"is missing columns:\n"
                    f"{missing_columns}"

                )


            caption_frames.append(

                df[
                    [
                        "filename",
                        "caption"
                    ]
                ].copy()

            )


        # ====================================================
        # COMBINE CAPTION FILES
        # ====================================================

        captions = pd.concat(

            caption_frames,

            ignore_index=True

        )


        # ====================================================
        # NORMALIZE FILENAMES
        # ====================================================

        captions["filename"] = (

            captions["filename"]

            .astype(str)

            .str.strip()

            .str.lower()

        )


        self.data["image"] = (

            self.data["image"]

            .astype(str)

            .str.strip()

            .str.lower()

        )


        # ====================================================
        # CLEAN CAPTIONS
        # ====================================================

        captions["caption"] = (

            captions["caption"]

            .fillna("")

            .astype(str)

            .str.strip()

        )


        # ====================================================
        # DUPLICATE CAPTION CHECK
        # ====================================================

        duplicate_count = (

            captions["filename"]

            .duplicated()

            .sum()

        )


        if duplicate_count > 0:

            print(

                f"[INFO] Duplicate caption filenames: "
                f"{duplicate_count}"

            )


        captions = (

            captions

            .drop_duplicates(

                subset=[
                    "filename"
                ],

                keep="first"

            )

        )


        # ====================================================
        # CREATE CAPTION MAP
        # ====================================================

        caption_map = (

            captions

            .set_index(
                "filename"
            )

            [
                "caption"
            ]

            .to_dict()

        )


        # ====================================================
        # MERGE
        # ====================================================

        self.data["text"] = (

            self.data["image"]

            .map(
                caption_map
            )

            .fillna("")

            .astype(str)

        )


        # ====================================================
        # STATISTICS
        # ====================================================

        total = len(
            self.data
        )


        with_caption = (

            self.data["text"]

            .str.strip()

            .ne("")

            .sum()

        )


        without_caption = (

            total
            -
            with_caption

        )


        print()

        print(

            f"[RESULT] Metadata images : "
            f"{total}"

        )


        print(

            f"[RESULT] Caption rows    : "
            f"{len(captions)}"

        )


        print(

            f"[RESULT] Images matched  : "
            f"{with_caption}"

        )


        print(

            f"[RESULT] Missing captions: "
            f"{without_caption}"

        )


        # ====================================================
        # SENSITIVE IMAGE CHECK
        # ====================================================

        if "subtype" in self.data.columns:

            sensitive_mask = (

                self.data["subtype"]

                .astype(str)

                .str.strip()

                .str.lower()

                .eq("sensitive")

            )


            sensitive_count = (

                sensitive_mask.sum()

            )


            sensitive_with_caption = (

                self.data.loc[
                    sensitive_mask,
                    "text"
                ]

                .str.strip()

                .ne("")

                .sum()

            )


            print()

            print(

                f"[RESULT] Sensitive images       : "
                f"{sensitive_count}"

            )


            print(

                f"[RESULT] Sensitive with captions: "
                f"{sensitive_with_caption}"

            )


        # ====================================================
        # SAMPLE
        # ====================================================

        print()

        print(
            "FIRST 10 CAPTION MAPPINGS"
        )

        print(
            "-" * 70
        )


        sample = self.data[
            [
                "image",
                "label",
                "text"
            ]
        ].head(10)


        print(

            sample.to_string(
                index=False
            )

        )


        print(
            "=" * 70
        )


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

    def _resolve_image_path(
        self,
        img_name
    ):

        img_str = (

            str(img_name)

            .strip()

        )


        # ----------------------------------------------------
        # Absolute path
        # ----------------------------------------------------

        if (

            os.path.isabs(
                img_str
            )

            and os.path.exists(
                img_str
            )

        ):

            return img_str


        # ----------------------------------------------------
        # Direct path
        # ----------------------------------------------------

        path1 = os.path.join(

            self.image_dir,

            img_str

        )


        if os.path.exists(
            path1
        ):

            return path1


        # ----------------------------------------------------
        # CSAM folder
        # ----------------------------------------------------

        path_csam = os.path.join(

            self.image_dir,

            "CSAM",

            os.path.basename(
                img_str
            )

        )


        if os.path.exists(
            path_csam
        ):

            return path_csam


        # ----------------------------------------------------
        # NON-CSAM folder
        # ----------------------------------------------------

        path_noncsam = os.path.join(

            self.image_dir,

            "nonCSAM",

            os.path.basename(
                img_str
            )

        )


        if os.path.exists(
            path_noncsam
        ):

            return path_noncsam


        # ----------------------------------------------------
        # NON_CSAM/images
        # ----------------------------------------------------

        path_non_csam = os.path.join(

            self.image_dir,

            "NON_CSAM",

            "images",

            os.path.basename(
                img_str
            )

        )


        if os.path.exists(
            path_non_csam
        ):

            return path_non_csam


        # ----------------------------------------------------
        # TRAIN / VAL / TEST paths
        # ----------------------------------------------------

        clean_path = (

            img_str

            .lstrip("/")

            .lstrip("\\")

        )


        dataset_path = os.path.join(

            self.image_dir,

            clean_path

        )


        if os.path.exists(
            dataset_path
        ):

            return dataset_path


        # ----------------------------------------------------
        # Search recursively as final fallback
        # ----------------------------------------------------

        filename = os.path.basename(
            img_str
        )


        for root, dirs, files in os.walk(
            self.image_dir
        ):

            if filename in files:

                return os.path.join(
                    root,
                    filename
                )


        # ----------------------------------------------------
        # Return fallback
        # ----------------------------------------------------

        return path1


# ============================================================
# LENGTH
# ============================================================

    def __len__(
        self
    ):

        return len(
            self.data
        )


# ============================================================
# GET ITEM
# ============================================================

    def __getitem__(
        self,
        idx
    ):

        row = self.data.iloc[
            idx
        ]


        # ----------------------------------------------------
        # Image name
        # ----------------------------------------------------

        image_name = row[
            "image"
        ]


        # ----------------------------------------------------
        # Caption / text
        # ----------------------------------------------------

        text = row[
            "text"
        ]


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = int(
            row["label"]
        )


        # ====================================================
        # IMAGE PATH
        # ====================================================

        img_path = None


        # ----------------------------------------------------
        # First preference:
        # metadata filepath
        # ----------------------------------------------------

        if (

            "filepath" in row.index

            and pd.notna(
                row["filepath"]
            )

        ):

            filepath = (

                str(
                    row["filepath"]
                )

                .strip()

            )


            # -----------------------------------------------
            # Absolute filepath
            # -----------------------------------------------

            if os.path.isabs(
                filepath
            ):

                if os.path.exists(
                    filepath
                ):

                    img_path = filepath


            # -----------------------------------------------
            # Relative filepath
            # -----------------------------------------------

            if img_path is None:

                filepath_clean = (

                    filepath

                    .lstrip("/")

                    .lstrip("\\")

                )


                candidate = os.path.join(

                    self.image_dir,

                    filepath_clean

                )


                if os.path.exists(
                    candidate
                ):

                    img_path = candidate


                # -------------------------------------------
                # Try filepath directly
                # -------------------------------------------

                if (

                    img_path is None

                    and os.path.exists(
                        filepath
                    )

                ):

                    img_path = filepath


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if img_path is None:

            img_path = (

                self._resolve_image_path(
                    image_name
                )

            )


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        try:

            image = (

                Image.open(
                    img_path
                )

                .convert(
                    "RGB"
                )

            )

        except Exception:

            print(
                "[WARNING] Could not load image:"
            )

            print(
                f"          {img_path}"
            )


            image = Image.new(

                "RGB",

                (
                    384,
                    384
                ),

                color=(
                    128,
                    128,
                    128
                )

            )


        # ====================================================
        # SIGLIP PROCESSING
        # ====================================================

        pixel_values = (

            self.processor(

                images=image,

                return_tensors="pt"

            )[

                "pixel_values"

            ]

            .squeeze(0)

        )


        # ====================================================
        # LOGIC VECTOR
        # ====================================================

        if self.has_logic_cols:

            logic_vec = torch.tensor(

                [

                    float(
                        row[
                            "nudity_score"
                        ]
                    ),

                    float(
                        row[
                            "p_minor_max"
                        ]
                    ),

                    float(
                        row[
                            "csam_init"
                        ]
                    ),

                    float(
                        row[
                            "adult_abuse"
                        ]
                    ),

                    float(
                        row[
                            "bully_score"
                        ]
                    )

                ],

                dtype=torch.float32

            )


        elif idx in self.logic_cache:

            logic_vec = (

                self.logic_cache[
                    idx
                ]

            )


        else:

            logic_vec = (

                compute_logic_vector_for_image(

                    img_path,

                    text

                )

            )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "image": pixel_values,

            "text": text,

            "logic_vector": logic_vec,

            "label": torch.tensor(

                label,

                dtype=torch.long

            ),

            "image_name": str(
                image_name
            )

        }