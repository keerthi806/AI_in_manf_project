from collections import defaultdict
from pathlib import Path

import pandas as pd


def get_df_filtered(data_path="AI_mfg_data"):
    data_path = Path(data_path)

    gt_paths_list = list((data_path / "gt_labels").glob("*.mat"))
    images_paths_list = list((data_path / "train_images").glob("*.png"))

    image_ids = sorted(
        set([int(id.name.split(".")[0].split("_")[-2]) for id in images_paths_list])
    )

    gt_path_dict = {
        Path(path).stem.split("_")[-1]: path
        for path in gt_paths_list
    }

    image_dict = defaultdict(dict)

    for path in images_paths_list:
        parts = path.stem.split("_")
        sample_id = parts[2]
        image_num = parts[3]
        image_dict[sample_id][image_num] = path

    df_all = pd.DataFrame({
        "ids": image_ids
    })

    df_all["label_path"] = df_all["ids"].astype(str).map(gt_path_dict)

    for i in [1, 2]:
        df_all[f"image_{i}"] = df_all["ids"].astype(str).map(
            lambda x: image_dict.get(x, {}).get(str(i))
        )

    df_all = df_all[["image_1", "image_2", "label_path"]]

    df_filtered = df_all[df_all["label_path"].notna()]

    return df_filtered
