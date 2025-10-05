"""cluster.py


from load_data import load_csv, preprocess
from utils import plot_elbow, compute_silhouette




def build_pipeline(n_clusters=5, random_state=42):
pipe = Pipeline([
("scaler", StandardScaler()),
("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
])
return pipe




def find_elbow(X, k_min=2, k_max=10):
inertias = []
ks = list(range(k_min, k_max + 1))
for k in ks:
km = KMeans(n_clusters=k, random_state=42)
km.fit(X)
inertias.append(km.inertia_)
return ks, inertias




def train(data_path: str, output_path: str, n_clusters=5):
df = load_csv(data_path)
from load_data import DEFAULT_FEATURES
X, df_clean = preprocess(df)


# elbow
ks, inertias = find_elbow(X, k_min=2, k_max=10)


# train final model
pipe = build_pipeline(n_clusters=n_clusters)
pipe.fit(X)
labels = pipe.named_steps["kmeans"].labels_


sil = compute_silhouette(X, labels)


# attach cluster labels to dataframe
df_clean = df_clean.reset_index(drop=True)
df_clean["cluster"] = labels


# save artifacts
joblib.dump(pipe, output_path)
# save a small diagnostic summary
summary = {
"n_samples": int(len(X)),
"n_features": int(X.shape[1]),
"n_clusters": int(n_clusters),
"silhouette_score": float(sil)
}
summary_path = Path(output_path).with_suffix('.summary.json')
import json
summary_path.write_text(json.dumps(summary, indent=2))


# optionally save elbow plot
plot_elbow(inertias, ks, out_path=str(Path(output_path).with_suffix('.elbow.png')))


print(f"Model saved to {output_path}")
print(f"Silhouette score: {sil:.4f}")




if __name__ == "__main__":
p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", default="kmeans_model.joblib")
p.add_argument("--n_clusters", type=int, default=5)
args = p.parse_args()
train(args.data, args.output, args.n_clusters)
