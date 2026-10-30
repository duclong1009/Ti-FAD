"""
Tạo setting chia lớp mới cho Ti-FAD (ActivityNet 1.3, 200 lớp: 100 train / 100 test)
sao cho embedding text (CLIP) của lớp train XA embedding của lớp test nhất.

Khác với THUMOS14: C(200,100) ~ 9e58 cách chia nên KHÔNG thể duyệt hết.
Thay vào đó dùng simulated annealing (hoán đổi 1 lớp test <-> 1 lớp train) với nhiều
lần khởi tạo ngẫu nhiên, rồi chọn K cách chia tốt nhất, khác nhau ít nhất --min-diff lớp test.

Điểm của một cách chia = trung bình, trên các lớp test, của cosine với lớp train GẦN NHẤT
(thấp hơn = xa hơn). Hoà điểm thì xét giá trị lớn nhất (trường hợp xấu nhất).

Tên lớp lấy từ (ưu tiên theo thứ tự):
  1. --classes-file  : mỗi dòng một tên lớp
  2. --anet-json     : activity_net.v1-3.min.json (lấy tất cả label trong annotations)
  3. --orig-root     : hợp train + test của split_0 gốc (đảm bảo trùng cách đặt tên của repo)

Chạy từ thư mục gốc của repo Ti-FAD:
  python make_far_splits_anet.py
  python make_far_splits_anet.py --anet-json path/to/activity_net.v1-3.min.json
  python make_far_splits_anet.py --num-splits 10 --min-diff 10 --restarts 30 --iters 60000
"""
import argparse
import json
import os

import numpy as np


# ----------------------------------------------------------------------------
# Tên lớp
# ----------------------------------------------------------------------------
def read_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_class_names(args):
    if args.classes_file:
        names = read_list(args.classes_file)
    elif args.anet_json:
        with open(args.anet_json) as f:
            data = json.load(f)
        db = data.get('database', {})
        names = sorted({a['label'] for v in db.values() for a in v.get('annotations', [])})
        if len(names) != 200 and 'taxonomy' in data:
            # dự phòng: lấy các nút lá của taxonomy (nút không là cha của nút nào)
            parents = {t['parentName'].strip().lower() for t in data['taxonomy']}
            leaves = {t['nodeName'] for t in data['taxonomy']
                      if t['nodeName'].strip().lower() not in parents}
            if len(leaves) == 200:
                names = sorted(leaves)
    else:
        tr = os.path.join(args.orig_root, 'train', 'split_0.list')
        te = os.path.join(args.orig_root, 'test', 'split_0.list')
        if not (os.path.exists(tr) and os.path.exists(te)):
            raise SystemExit(
                "Không tìm thấy tên lớp. Hãy cung cấp --classes-file, --anet-json "
                "hoặc đúng --orig-root (chứa train/split_0.list và test/split_0.list).")
        names = sorted(set(read_list(tr)) | set(read_list(te)))
    names = list(dict.fromkeys(names))  # bỏ trùng, giữ thứ tự
    if len(names) != 200:
        print(f"[CẢNH BÁO] ActivityNet 1.3 có 200 lớp nhưng đọc được {len(names)} lớp.")
    return names


# ----------------------------------------------------------------------------
# Embedding CLIP
# ----------------------------------------------------------------------------
def clip_text_embeddings(names, model_path, prompt, device, batch=64):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    texts = [prompt.format(c) for c in names]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            inputs = tok(texts[i:i + batch], padding=True, return_tensors="pt").to(device)
            out.append(model.get_text_features(**inputs))
    emb = torch.cat(out)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


# ----------------------------------------------------------------------------
# Chấm điểm và tìm kiếm
# ----------------------------------------------------------------------------
def split_score(sim, test_idx, train_idx):
    """(trung bình, lớn nhất) của cosine từ mỗi lớp test tới lớp train gần nhất."""
    nearest = sim[np.ix_(test_idx, train_idx)].max(axis=1)
    return float(nearest.mean()), float(nearest.max())


def anneal(sim, n_test, iters, t0, t1, rng):
    """Simulated annealing: hoán đổi 1 lớp test với 1 lớp train, giảm điểm trung bình."""
    n = sim.shape[0]
    perm = rng.permutation(n)
    test, train = perm[:n_test].copy(), perm[n_test:].copy()
    cur = sim[np.ix_(test, train)].max(axis=1).mean()
    best, best_test = cur, test.copy()

    a_all = rng.integers(0, n_test, size=iters)
    b_all = rng.integers(0, n - n_test, size=iters)
    u_all = rng.random(iters)
    for it in range(iters):
        temp = t0 * (t1 / t0) ** (it / iters)
        a, b = a_all[it], b_all[it]
        test[a], train[b] = train[b], test[a]
        new = sim[np.ix_(test, train)].max(axis=1).mean()
        d = new - cur
        if d < 0 or u_all[it] < np.exp(-d / temp):
            cur = new
            if cur < best:
                best, best_test = cur, test.copy()
        else:
            test[a], train[b] = train[b], test[a]   # hoàn tác
    return np.sort(best_test)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='openai/clip-vit-base-patch16')
    p.add_argument('--prompt', default='{}',
                   help="chuỗi đưa vào CLIP; Ti-FAD dùng tên lớp gốc nên mặc định là '{}'")
    p.add_argument('--device', default='cpu')
    p.add_argument('--classes-file', default=None)
    p.add_argument('--anet-json', default=None)
    p.add_argument('--num-test', type=int, default=100)
    p.add_argument('--num-splits', type=int, default=10)
    p.add_argument('--min-diff', type=int, default=10,
                   help='hai split được chọn phải khác nhau ít nhất bấy nhiêu lớp test')
    p.add_argument('--restarts', type=int, default=30, help='số lần chạy annealing độc lập')
    p.add_argument('--iters', type=int, default=60000, help='số bước hoán đổi mỗi lần chạy')
    p.add_argument('--t0', type=float, default=2e-3, help='nhiệt độ đầu')
    p.add_argument('--t1', type=float, default=1e-5, help='nhiệt độ cuối')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--random-baseline', type=int, default=2000,
                   help='số split ngẫu nhiên dùng để ước lượng xếp hạng')
    p.add_argument('--out-root', default='splits/train_50_test_50_far/ActivityNet1.3')
    p.add_argument('--orig-root', default='splits/train_50_test_50/ActivityNet1.3',
                   help='split gốc để so sánh (bỏ qua nếu không có)')
    args = p.parse_args()

    names = load_class_names(args)
    n = len(names)
    print(f"{n} lớp, {args.num_test} test / {n - args.num_test} train")

    emb = clip_text_embeddings(names, args.model, args.prompt, args.device)
    sim = emb @ emb.T
    rng = np.random.default_rng(args.seed)

    # mốc so sánh: split ngẫu nhiên
    rand_scores = np.empty(args.random_baseline)
    for i in range(args.random_baseline):
        perm = rng.permutation(n)
        rand_scores[i] = split_score(sim, perm[:args.num_test], perm[args.num_test:])[0]
    print(f"Split ngẫu nhiên ({args.random_baseline}): min {rand_scores.min():.4f} | "
          f"trung vị {np.median(rand_scores):.4f} | max {rand_scores.max():.4f}")

    # tìm kiếm
    cands = []
    for r in range(args.restarts):
        test = anneal(sim, args.num_test, args.iters, args.t0, args.t1, rng)
        train = np.setdiff1d(np.arange(n), test)
        m, w = split_score(sim, test, train)
        cands.append((m, w, test))
        print(f"  restart {r + 1:2d}/{args.restarts}: mean={m:.4f} worst={w:.4f}")
    cands.sort(key=lambda c: (c[0], c[1]))

    # chọn K split khác nhau đủ xa
    chosen = []
    for m, w, test in cands:
        ts = set(test.tolist())
        if all(len(ts - set(c[2].tolist())) >= args.min_diff for c in chosen):
            chosen.append((m, w, test))
        if len(chosen) == args.num_splits:
            break
    if len(chosen) < args.num_splits:
        print(f"[CẢNH BÁO] chỉ chọn được {len(chosen)}/{args.num_splits} split; "
              f"hãy tăng --restarts hoặc giảm --min-diff.")

    for sub in ('train', 'test'):
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=True)

    print("\nCác split far:")
    for i, (m, w, test) in enumerate(chosen):
        train = np.setdiff1d(np.arange(n), test)
        for sub, idx in (('train', train), ('test', test)):
            with open(os.path.join(args.out_root, sub, f'split_{i}.list'), 'w') as f:
                f.write('\n'.join(names[j] for j in idx) + '\n')
        pct = 100 * np.mean(rand_scores <= m)
        print(f"[far split_{i}] mean={m:.4f} worst={w:.4f}  (xa hơn {100 - pct:.1f}% split ngẫu nhiên)")

    # so sánh với các split gốc
    if os.path.isdir(args.orig_root):
        print("\nSplit gốc (để so sánh):")
        idx_of = {c: i for i, c in enumerate(names)}
        for i in range(10):
            ft = os.path.join(args.orig_root, 'test', f'split_{i}.list')
            fr = os.path.join(args.orig_root, 'train', f'split_{i}.list')
            if not (os.path.exists(ft) and os.path.exists(fr)):
                continue
            try:
                t = [idx_of[c] for c in read_list(ft)]
                r_ = [idx_of[c] for c in read_list(fr)]
            except KeyError as e:
                print(f"[orig split_{i}] tên lớp không khớp: {e}")
                continue
            m, w = split_score(sim, t, r_)
            pct = 100 * np.mean(rand_scores <= m)
            print(f"[orig split_{i}] mean={m:.4f} worst={w:.4f}  (xa hơn {100 - pct:.1f}% split ngẫu nhiên)")

    print(f"\nĐã ghi file vào {args.out_root}/{{train,test}}/split_*.list")


if __name__ == '__main__':
    main()