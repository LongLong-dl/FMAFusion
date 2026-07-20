import os
import csv
import random
import argparse

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_ROOT = os.path.join(_PROJ_ROOT, 'dataset', 'nutrition5k_dataset')
_DEFAULT_OUT_DIR = os.path.join(_DEFAULT_DATA_ROOT, 'imagery')


def parse_args():
    parser = argparse.ArgumentParser(description='Split Nutrition5K dataset into train/test sets (70/30)')

    parser.add_argument('--data_root', type=str, default=_DEFAULT_DATA_ROOT,
                        help='path to nutrition5k_dataset root')
    parser.add_argument('--train_ratio', type=float, default=0.8333,
                        help='train ratio (default: 0.8333). test ratio = 1 - train')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible split')
    parser.add_argument('--out_dir', type=str, default=_DEFAULT_OUT_DIR,
                        help='output directory for split CSV files')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help='output filename for train set')
    parser.add_argument('--test_csv', type=str, default='test.csv',
                        help='output filename for test set')

    args = parser.parse_args()
    assert 0 < args.train_ratio < 1, 'train_ratio must be between 0 and 1'
    return args


def load_nutrition_values(csv_path):
    """Load dish_nutrition_values.csv, return dict: dish_id -> (calories, mass, fat, carb, protein)"""
    nutrition = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row['dish_id']
            nutrition[dish_id] = (
                float(row['calories']),
                float(row['mass']),
                float(row['fat']),
                float(row['carb']),
                float(row['protein']),
            )
    return nutrition


def find_available_dishes(imagery_dir):
    """Find dish directories that contain both rgb.png and depth_color.png."""
    dishes = []
    for d in os.listdir(imagery_dir):
        dish_path = os.path.join(imagery_dir, d)
        if not os.path.isdir(dish_path):
            continue
        if os.path.exists(os.path.join(dish_path, 'rgb.png')) and \
           os.path.exists(os.path.join(dish_path, 'depth_color.png')):
            dishes.append(d)
    return sorted(dishes)


def write_split_csv(out_path, dish_ids, nutrition):
    """Write a CSV file with dish_id and nutrition columns for the given dish_ids."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dish_id', 'calories', 'mass', 'fat', 'carb', 'protein'])
        for dish_id in dish_ids:
            cal, mass, fat, carb, protein = nutrition[dish_id]
            writer.writerow([dish_id, cal, mass, fat, carb, protein])


def main():
    args = parse_args()

    imagery_dir = os.path.join(args.data_root, 'imagery', 'realsense_overhead')
    csv_path = os.path.join(args.data_root, 'dish_nutrition_values.csv')

    if not os.path.exists(imagery_dir):
        raise FileNotFoundError(f'Imagery directory not found: {imagery_dir}')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Nutrition CSV not found: {csv_path}')

    nutrition = load_nutrition_values(csv_path)
    print(f'Loaded nutrition values for {len(nutrition)} dishes')

    all_dishes = find_available_dishes(imagery_dir)
    available_dishes = [d for d in all_dishes if d in nutrition]
    missing_labels = [d for d in all_dishes if d not in nutrition]

    print(f'Dishes with images: {len(all_dishes)}')
    print(f'Dishes with images + labels: {len(available_dishes)}')
    if missing_labels:
        print(f'Dishes with images but NO labels (skipped): {len(missing_labels)}')

    random.seed(args.seed)
    random.shuffle(available_dishes)

    n_train = int(len(available_dishes) * args.train_ratio)
    train_dishes = available_dishes[:n_train]
    test_dishes = available_dishes[n_train:]

    print(f'Train set: {len(train_dishes)} dishes')
    print(f'Test set:  {len(test_dishes)} dishes')
    print(f'Train ratio: {len(train_dishes) / len(available_dishes) * 100:.1f}%')
    print(f'Test ratio:  {len(test_dishes) / len(available_dishes) * 100:.1f}%')

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, args.train_csv)
    test_path = os.path.join(args.out_dir, args.test_csv)

    write_split_csv(train_path, train_dishes, nutrition)
    print(f'Written {len(train_dishes)} rows to {train_path}')

    write_split_csv(test_path, test_dishes, nutrition)
    print(f'Written {len(test_dishes)} rows to {test_path}')

    print(f'\nDataset split complete. Seed: {args.seed}')


if __name__ == '__main__':
    main()
