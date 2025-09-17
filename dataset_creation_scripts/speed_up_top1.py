import h5py, numpy as np, os

SRC = "/home/omiqran/metamaterials_urop/new_top1_cnn.h5"
DST = "/home/omiqran/metamaterials_urop/new_top1_cnn_fast.h5"

def per_sample_chunks(shape):
    # pattern shapes: (N,H,W) or (N,1,H,W)
    if len(shape) == 3:          # N,H,W
        return (1, shape[1], shape[2])
    elif len(shape) == 4:        # N,C,H,W
        return (1, shape[1], shape[2], shape[3])
    else:
        raise ValueError(f"Unexpected pattern shape {shape}")

with h5py.File(SRC, "r") as fi, h5py.File(DST, "w") as fo:
    for split in ("train", "test"):
        # --- pattern ---
        if f"pattern_{split}" in fi:
            p = fi[f"pattern_{split}"]
            ch = per_sample_chunks(p.shape)
            fo.create_dataset(
                f"pattern_{split}",
                shape=p.shape,
                dtype=p.dtype,
                compression="lzf",   # or compression=None
                shuffle=True,
                chunks=ch,
            )
            # write in batches to keep memory reasonable
            bs = 1024
            N = p.shape[0]
            for s in range(0, N, bs):
                e = min(N, s + bs)
                fo[f"pattern_{split}"][s:e] = p[s:e]

        # --- params ---
        if f"params_{split}" in fi:
            q = fi[f"params_{split}"]
            N, P = q.shape
            fo.create_dataset(
                f"params_{split}",
                shape=q.shape,
                dtype=q.dtype,
                compression="lzf",
                shuffle=True,
                chunks=(min(2048, N), P),
            )
            fo[f"params_{split}"][:] = q[...]

        # --- scalars ---
        for key in (f"neff_{split}", f"weight_{split}"):
            if key in fi:
                r = fi[key]
                N = r.shape[0]
                fo.create_dataset(
                    key,
                    shape=r.shape,
                    dtype=r.dtype,
                    compression="lzf",
                    shuffle=True,
                    chunks=(min(8192, N),),
                )
                fo[key][:] = r[...]

    # copy meta attrs if present
    if "meta" in fi:
        m = fo.create_group("meta")
        for k, v in fi["meta"].attrs.items():
            m.attrs[k] = v

print("Wrote:", DST)
