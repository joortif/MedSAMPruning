import pandas as pd

def fuse_cb_el2n(cb_dict, el2n_dict, alpha, csv_path):

    if alpha is None:
        print("Alpha is None, using default value of 0.5")
        alpha = 0.5

    df = pd.DataFrame({
        "id": sorted(set(cb_dict.keys()) | set(el2n_dict.keys()))
    })

    df["cb_scs"] = df["id"].map(cb_dict)
    df["el2n"] = df["id"].map(el2n_dict)

    df["cb_scs_norm"] = df["cb_scs"].rank(pct=True)
    df["el2n_norm"] = df["el2n"].rank(pct=True)

    df["fusion"] = (
        alpha * df["cb_scs_norm"]
        + (1 - alpha) * df["el2n_norm"]
    )

    output = df[["id", "fusion"]].sort_values(
        by="fusion",
        ascending=False
    )

    output.to_csv(csv_path,
        index=False,
        sep=";",
        decimal=",")
    
    return dict(zip(df["id"], df["fusion"]))