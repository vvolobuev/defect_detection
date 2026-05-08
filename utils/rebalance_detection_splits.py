import argparse
import random
import shutil
from pathlib import Path


def rebalance_dataset(ds, seed):
    train_img = ds / "train" / "images"
    train_lbl = ds / "train" / "labels"
    valid_img = ds / "valid" / "images"
    valid_lbl = ds / "valid" / "labels"
    test_img = ds / "test" / "images"
    test_lbl = ds / "test" / "labels"

    for p in [valid_img, valid_lbl, test_img, test_lbl]:
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*"):
            f.unlink()

    img_map = {p.stem: p for p in train_img.glob("*.png")}
    lbl_map = {p.stem: p for p in train_lbl.glob("*.txt")}

                                                
    for s in sorted(set(img_map) - set(lbl_map)):
        img_map[s].unlink(missing_ok=True)
    for s in sorted(set(lbl_map) - set(img_map)):
        lbl_map[s].unlink(missing_ok=True)

    stems = sorted(set(img_map) & set(lbl_map))
    random.Random(seed).shuffle(stems)

    n = len(stems)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.2)
    valid_stems = stems[n_train:n_train + n_valid]
    test_stems = stems[n_train + n_valid:]

    def move_many(stems_list, dst_i, dst_l):
        for s in stems_list:
            src_i = train_img / f"{s}.png"
            src_l = train_lbl / f"{s}.txt"
            if src_i.exists():
                shutil.move(str(src_i), str(dst_i / src_i.name))
            if src_l.exists():
                shutil.move(str(src_l), str(dst_l / src_l.name))

    move_many(valid_stems, valid_img, valid_lbl)
    move_many(test_stems, test_img, test_lbl)

                                               
    for split in ["train", "valid", "test"]:
        i_dir = ds / split / "images"
        l_dir = ds / split / "labels"
        i_set = {p.stem for p in i_dir.glob("*.png")}
        l_set = {p.stem for p in l_dir.glob("*.txt")}
        for s in sorted(i_set - l_set):
            (i_dir / f"{s}.png").unlink(missing_ok=True)
        for s in sorted(l_set - i_set):
            (l_dir / f"{s}.txt").unlink(missing_ok=True)

    result = {}
    for split in ["train", "valid", "test"]:
        i_count = sum(1 for _ in (ds / split / "images").glob("*.png"))
        l_count = sum(1 for _ in (ds / split / "labels").glob("*.txt"))
        result[split] = (i_count, l_count)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=177)
    args = parser.parse_args()

    for ds_name in ["defects_rgb_det", "defects_depth_heat_det"]:
        ds = args.datasets_root / ds_name
        if not ds.exists():
            print(f"[skip] {ds} not found")
            continue
        stats = rebalance_dataset(ds, seed=args.seed)
        print(ds_name)
        for split in ["train", "valid", "test"]:
            i_count, l_count = stats[split]
            print(f"{split}: images={i_count} labels={l_count}")


if __name__ == "__main__":
    main()
