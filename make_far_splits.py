"""
Tạo setting chia lớp mới cho Ti-FAD (THUMOS14, 10 train / 10 test) sao cho
embedding text (CLIP) của lớp train XA embedding của lớp test nhất.

Cách làm:
  1. Mã hoá 20 tên lớp bằng đúng text encoder Ti-FAD dùng
     (openai/clip-vit-base-patch16, đưa thẳng tên lớp vào như TextFeatures.forward).
  2. Duyệt TẤT CẢ C(20,10) = 184,756 cách chia.
  3. Với mỗi cách chia, mỗi lớp test lấy độ tương đồng cosine với lớp train GẦN NHẤT.
     Điểm của cách chia = trung bình các giá trị đó (thấp hơn = xa hơn).
     Hoà điểm thì xét giá trị lớn nhất (trường hợp xấu nhất).
  4. Chọn ra K cách chia tốt nhất, khác nhau ít nhất --min-diff lớp test,
     ghi thành splits/train_50_test_50_far/THUMOS14/{train,test}/split_k.list

Chạy từ thư mục gốc của repo Ti-FAD:
  python make_far_splits.py                 # 10 split, mặc định
  python make_far_splits.py --num-splits 5 --min-diff 3
"""
import argparse
import itertools
import os

import numpy as np

THUMOS_CLASSES = [
    'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
    'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
    'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
    'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
    'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking',
]


def clip_text_embeddings(names, model_path, prompt):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).eval()
    texts = [prompt.format(c) for c in names]
    with torch.no_grad():
        inputs = tok(texts, padding=True, return_tensors="pt")
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def score_partitions(sim, n_test):
    """Trả về (danh sách tập test, điểm trung bình, điểm xấu nhất) cho mọi cách chia."""
    n = sim.shape[0]
    all_idx = np.arange(n)
    combos = np.array(list(itertools.combinations(range(n), n_test)))   # (M, n_test)
    is_test = np.zeros((len(combos), n), dtype=bool)
    is_test[np.arange(len(combos))[:, None], combos] = True

    # sim của từng lớp test với mọi lớp: (M, n_test, n)
    s = sim[combos]
    # che các cột là lớp test -> chỉ giữ lớp train
    s = np.where(is_test[:, None, :], -np.inf, s)
    nearest = s.max(axis=2)                     # (M, n_test): lớp train gần nhất
    return combos, nearest.mean(axis=1), nearest.max(axis=1)


def split_score(sim, test_idx):
    n = sim.shape[0]
    train_idx = [i for i in range(n) if i not in set(test_idx)]
    nearest = sim[np.ix_(test_idx, train_idx)].max(axis=1)
    return nearest.mean(), nearest.max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='openai/clip-vit-base-patch16')
    p.add_argument('--prompt', default='{}',
                   help="chuỗi đưa vào CLIP; Ti-FAD dùng tên lớp gốc nên mặc định là '{}'")
    p.add_argument('--num-test', type=int, default=10)
    p.add_argument('--num-splits', type=int, default=10)
    p.add_argument('--min-diff', type=int, default=2,
                   help='hai split được chọn phải khác nhau ít nhất bấy nhiêu lớp test')
    p.add_argument('--out-root', default='splits/train_50_test_50_far/THUMOS14')
    p.add_argument('--orig-root', default='splits/train_50_test_50/THUMOS14',
                   help='split gốc để so sánh (bỏ qua nếu không có)')
    args = p.parse_args()

    names = THUMOS_CLASSES
    emb = clip_text_embeddings(names, args.model, args.prompt)
    sim = emb @ emb.T

    combos, mean_s, max_s = score_partitions(sim, args.num_test)
    order = np.lexsort((max_s, mean_s))          # tăng dần theo mean, rồi max

    chosen = []
    for k in order:
        test_set = set(combos[k])
        if all(len(test_set - set(combos[c])) >= args.min_diff for c in chosen):
            chosen.append(k)
        if len(chosen) == args.num_splits:
            break

    for sub in ('train', 'test'):
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=True)

    print(f"Đã xét {len(combos):,} cách chia. Điểm = cosine trung bình (lớp test -> lớp train gần nhất), thấp = xa.")
    print(f"Toàn bộ: min {mean_s.min():.4f} | trung vị {np.median(mean_s):.4f} | max {mean_s.max():.4f}\n")

    for i, k in enumerate(chosen):
        test_idx = sorted(combos[k])
        train_idx = [j for j in range(len(names)) if j not in test_idx]
        for sub, idx in (('train', train_idx), ('test', test_idx)):
            with open(os.path.join(args.out_root, sub, f'split_{i}.list'), 'w') as f:
                f.write('\n'.join(names[j] for j in idx) + '\n')
        print(f"[far split_{i}] mean={mean_s[k]:.4f} worst={max_s[k]:.4f}")
        print("   test :", ', '.join(names[j] for j in test_idx))

    # so sánh với 10 split gốc
    if os.path.isdir(args.orig_root):
        print("\nSplit gốc (để so sánh):")
        rank = {tuple(c): r for r, c in enumerate(combos[order])}
        for i in range(10):
            fpath = os.path.join(args.orig_root, 'test', f'split_{i}.list')
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                t = [names.index(l.strip()) for l in f if l.strip()]
            m, w = split_score(sim, t)
            pct = 100 * rank[tuple(sorted(t))] / len(combos)
            print(f"[orig split_{i}] mean={m:.4f} worst={w:.4f}  (xếp hạng top {pct:.1f}%)")

    print(f"\nĐã ghi file vào {args.out_root}/{{train,test}}/split_*.list")


if __name__ == '__main__':
    main()