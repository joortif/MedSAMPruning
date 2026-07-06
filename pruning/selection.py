from pathlib import Path
import shutil

def select_samples(
    scores_dict,
    pruning_rate,
    strategy="hard",
    difficulty_high_score=True,
    mixing_percentage=0.5,
):

    N = len(scores_dict)

    n_keep = int(round(N * (1 - pruning_rate)))
    print(f"Samples to keep: {n_keep} out of {N}")

    items = list(scores_dict.items())

    items.sort(
        key=lambda x: x[1],
        reverse=difficulty_high_score
    )

    if strategy == "hard":
        selected = items[:n_keep]

    elif strategy == "easy":
        selected = items[-n_keep:]

    elif strategy == "mixed":

        n_hard = int(round(n_keep * mixing_percentage))
        n_easy = n_keep - n_hard

        hard_samples = items[:n_hard]
        easy_samples = items[-n_easy:]

        selected = hard_samples + easy_samples

        seen = set()
        selected = [
            item for item in selected
            if not (item[0] in seen or seen.add(item[0]))
        ]

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Use 'hard', 'easy' or 'mixed'."
        )

    return [img_id for img_id, _ in selected]

def save_pruned_samples(
    selected_ids,
    images_dir,
    labels_dir,
    output_dir,
):

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file()
    }

    label_files = {
        p.stem: p
        for p in labels_dir.iterdir()
        if p.is_file()
    }

    copied = 0
    
    for img_id in selected_ids:
        
        img_id = Path(str(img_id)).stem
        
        img_path = image_files.get(img_id)
        label_path = label_files.get(img_id)

        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)

        copied += 1

    print(f"Copied {copied} samples to {output_dir}")