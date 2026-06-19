import pandas as pd

def cb_scs_el2n_fusion(files, alpha, out_dir=None):

        metric1 = pd.read_csv(files[0], sep=";")
        metric2 = pd.read_csv(files[1], sep=";")

        score1 = metric1.columns[1]
        score2 = metric2.columns[1]

        df = metric1.merge(metric2, on="id", how="inner")

        df[f"{score1}_norm"] = df[score1].rank(pct=True)
        df[f"{score2}_norm"] = df[score2].rank(pct=True)


        df["fusion_score"] = (
            alpha * df[f"{score1}_norm"]
            + (1 - alpha) * df[f"{score2}_norm"]
        )

        output = df[["id", "fusion_score"]]
        output = output.sort_values(by="fusion_score", ascending=False)
        if out_dir is not None:
                output.to_csv(f"{out_dir}/{score1}_{score2}_{alpha}.csv", sep=";", index=False)