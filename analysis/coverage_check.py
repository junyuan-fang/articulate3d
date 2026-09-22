import numpy as np, glob, os, json, pickle
d = '/data/challenge/USDNet/data/processed/articulate3d_challenge_mov/instance_gt/validation'
tot = 0
per = {}
for f in sorted(glob.glob(d + '/*.txt')):
    ids = np.loadtxt(f, dtype=np.int64)
    u = [i for i in np.unique(ids) if i // 1000 in (1, 2)]
    per[os.path.basename(f)[:-4]] = len(u)
    tot += len(u)
print('42-scene GT instances total:', tot)
print('a980334473 GT instances:', per.get('a980334473'))
pr = pickle.load(open('/data/challenge/USDNet/results/eval_mov_val/debug/val_preds/preds.pkl', 'rb'))
print('scenes in preds:', len(pr), 'missing:', sorted(set(per) - set(pr)))
cnt = {1: 0, 2: 0}
for s in pr:
    ids = np.loadtxt(d + '/%s.txt' % s, dtype=np.int64)
    for i in np.unique(ids):
        if i // 1000 in (1, 2):
            cnt[i // 1000] += 1
print('GT over 41 scenes: rotation=%d translation=%d total=%d' % (cnt[1], cnt[2], cnt[1] + cnt[2]))
r = json.load(open('/data/challenge/analysis/arti_error_diagnosis.json'))
rows = r['rows']
print('matched: rot=%d trans=%d' % (sum(x['label'] == 'rotation' for x in rows),
                                    sum(x['label'] == 'translation' for x in rows)))
