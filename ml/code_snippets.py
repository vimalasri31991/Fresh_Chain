"""
Code shown on the model pages.  Every snippet mirrors the real code in
the matching ml/*.py file, so what the page shows is what actually ran.
"""

_SPLIT = """# 80 / 20 split, same random_state for every model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)"""


def _model(title, tag, body):
    return {"title": title, "tag": tag, "code": body.strip("\n")}


SNIPPETS = {

    # ---------------- RANDOM FOREST ----------------
    "random-forest": [
        _model("Split the data", "Setup", _SPLIT),
        _model("Random Forest · Before tuning", "Before", """
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)"""),
        _model("Random Forest · After tuning", "After", """
model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)"""),
    ],

    # ---------------- ADABOOST ----------------
    "ada-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("AdaBoost · Before tuning", "Before", """
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
model.fit(X_train, y_train)"""),
        _model("AdaBoost · After tuning", "After", """
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- GRADIENT BOOST ----------------
    "gradient-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("Gradient Boosting · Before tuning", "Before", """
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)"""),
        _model("Gradient Boosting · After tuning", "After", """
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- XGBOOST ----------------
    "xg-boost": [
        _model("Split the data", "Setup", _SPLIT),
        _model("XGBoost · Before tuning", "Before", """
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)"""),
        _model("XGBoost · After tuning", "After", """
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- LIGHTGBM ----------------
    "lightgbm": [
        _model("Split the data", "Setup", _SPLIT),
        _model("LightGBM · Before tuning", "Before", """
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)"""),
        _model("LightGBM · After tuning", "After", """
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=20,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- PRUNING (4 sections) ----------------
    "pruning-fully-grown": [
        _model("Fully Grown Tree", "No pruning", """
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(model.get_depth(), model.get_n_leaves())"""),
    ],

    "pruning-pre-pruning": [
        _model("Pre-Pruning (stop early)", "Limits while growing", """
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)"""),
    ],

    "pruning-post-pruning": [
        _model("Post-Pruning (reduced-error)", "Prune after growing", """
# 1) grow a FULL tree on 75% of the training data
X_grow, X_val, y_grow, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=42, stratify=y_train
)
grown = DecisionTreeClassifier(random_state=42).fit(X_grow, y_grow)

# 2) bottom-up: replace a sub-tree by a leaf when that does
#    not increase the error on the validation part
def prune(node, idx):
    leaf_err = (y_val[idx] != node_class[node]).sum()
    if left[node] == -1:
        return leaf_err
    go_left = X_val[idx, feature[node]] <= threshold[node]
    sub_err = prune(left[node], idx[go_left]) + \\
              prune(right[node], idx[~go_left])
    if leaf_err <= sub_err:
        pruned.add(node)          # collapse this branch
        return leaf_err
    return sub_err"""),
    ],

    "pruning-cost-complexity": [
        _model("Cost-Complexity Pruning (ccp_alpha)", "Alpha chosen by CV", """
path = DecisionTreeClassifier(random_state=42) \\
           .cost_complexity_pruning_path(X_train, y_train)

# 5-fold CV accuracy for every candidate alpha (train data only)
scores = [
    cross_val_score(
        DecisionTreeClassifier(random_state=42, ccp_alpha=a),
        X_train, y_train, cv=5
    ).mean()
    for a in path.ccp_alphas
]

# one-standard-error rule: the simplest tree that is still
# within 1 SE of the best cross-validated score
best_alpha = max(a for a, s in zip(path.ccp_alphas, scores)
                 if s >= max(scores) - se)

model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
model.fit(X_train, y_train)"""),
    ],

    # ---------------- HIERARCHICAL CLUSTERING ----------------
    "hierarchical": [
        _model("SciPy Dendrogram & Cophenetic Correlation", "Dendrogram", """
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

Z_ward = linkage(X_sample, method="ward")
coph_corr, _ = cophenet(Z_ward, pdist(X_sample))
cut_height_k2 = (Z_ward[-1, 2] + Z_ward[-2, 2]) / 2.0

dendrogram(
    Z_ward,
    truncate_mode="lastp",
    p=30,
    color_threshold=cut_height_k2,
    show_contracted=True
)"""),
        _model("Agglomerative Clustering (Ward Linkage)", "Clustering", """
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

model = AgglomerativeClustering(n_clusters=2, linkage="ward")
labels = model.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)"""),
    ],

    # ---------------- DBSCAN ----------------
    "dbscan": [
        _model("DBSCAN Density Clustering & Noise Detection", "DBSCAN", """
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# 1) k-distance curve for selecting epsilon
nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
k_distances = np.sort(nn.kneighbors(X_scaled)[0][:, -1])

# 2) Fit DBSCAN and separate core clusters from noise (-1)
model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X_scaled)
noise_count = int((labels == -1).sum())"""),
    ],

    # ---------------- ANOMALY DETECTION ----------------
    "anomaly-detection": [
        _model("Isolation Forest · Anomaly Detection", "Primary Model", """
from sklearn.ensemble import IsolationForest

iso_model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)
iso_pred = iso_model.fit_predict(X_scaled)          # 1 = Normal, -1 = Anomaly
iso_scores = iso_model.decision_function(X_scaled)  # negative = Outlier"""),
        _model("LOF, One-Class SVM & Multi-Model Consensus", "Ensemble Consensus", """
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

lof_pred = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(X_scaled)
ocsvm_pred = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_train).predict(X_scaled)
elliptic_pred = EllipticEnvelope(contamination=0.05, random_state=42).fit_predict(X_scaled)

# Flag consensus anomalies detected by >= 2 independent detectors
votes = (iso_pred == -1).astype(int) + (lof_pred == -1).astype(int) + \\
        (ocsvm_pred == -1).astype(int) + (elliptic_pred == -1).astype(int)"""),
    ],

    # ---------------- PCA ----------------
    "pca": [
        _model("Principal Component Analysis & Explained Variance", "Eigen Decomposition", """
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df[FEATURES])

pca_full = PCA(n_components=4, random_state=42)
X_pca = pca_full.fit_transform(X_scaled)

explained_pct = pca_full.explained_variance_ratio_ * 100
cumulative_pct = np.cumsum(explained_pct)
loadings = pd.DataFrame(pca_full.components_.T, index=FEATURES, columns=["PC1", "PC2", "PC3", "PC4"])"""),
        _model("2D Dimensionality Reduction & Reconstruction Error", "Projection", """
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

X_reconstructed = pca_2d.inverse_transform(X_pca_2d)
recon_mse = mean_squared_error(X_scaled, X_reconstructed)"""),
    ],

    # ---------------- T-SNE ----------------
    "tsne": [
        _model("t-SNE Non-Linear Manifold Embedding", "Manifold Learning", """
from sklearn.manifold import TSNE, trustworthiness

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=750,
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)

kl_div = tsne.kl_divergence_
trust_score = trustworthiness(X_scaled, X_tsne, n_neighbors=10)"""),
    ],

    # ---------------- UMAP ----------------
    "umap": [
        _model("UMAP Fuzzy Simplicial Set Projection", "Topological Embedding", """
import umap
from sklearn.manifold import trustworthiness

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.10,
    metric="euclidean",
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)
trust_score = trustworthiness(X_scaled, X_umap, n_neighbors=10)"""),
    ],

    # ---------------- MODEL EVALUATION, SELECTION & CALIBRATION ----------------
    "model-evaluation": [
        _model("1. Three-Way Split & Lockbox Discipline", "Split Discipline", """
from sklearn.model_selection import train_test_split

# 70% Train, 15% Validation (for hyperparameter tuning & threshold choice), 15% Test (LOCKED)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
# Tune models strictly using X_val. Evaluate chosen model once on X_test!"""),

        _model("2. Stratified K-Fold & Nested Cross-Validation", "Nested CV", """
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Outer loop evaluates unbiased risk; Inner loop tunes hyperparameters
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    grid = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring="accuracy")
    grid.fit(X_tr, y_tr)
    nested_scores.append(grid.score(X_te, y_te))"""),

        _model("3. Classification Metrics & Class Imbalance", "Classification", """
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

cm = confusion_matrix(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
# Under heavy class imbalance, Accuracy lies and ROC-AUC is overoptimistic.
# PR-AUC directly reveals false alarm degradation!"""),

        _model("4. Regression Metrics & Outlier Sensitivity (L1 vs L2)", "Regression", """
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # L2 quadratic penalty: heavily penalizes outliers
mae = mean_absolute_error(y_true, y_pred)           # L1 linear penalty: robust against sensor spikes
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
r2 = r2_score(y_true, y_pred)"""),

        _model("5. Probability Calibration (Platt Scaling & Isotonic)", "Calibration", """
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Platt scaling (logistic sigmoid) or Isotonic regression (non-parametric monotonic step)
platt_model = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=3)
platt_model.fit(X_train, y_train)

iso_model = CalibratedClassifierCV(base_estimator, method="isotonic", cv=3)
iso_model.fit(X_train, y_train)

# Measure calibration accuracy with Brier Score Loss & Reliability Diagram
brier = brier_score_loss(y_test, platt_model.predict_proba(X_test)[:, 1])
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)"""),

        _model("6. Hyperparameter Search (Grid vs Random vs Bayes GP)", "Search Strategies", """
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Random search tests more distinct values per dominant dimension (Bergstra & Bengio)
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=20, cv=3)

# Bayesian Optimisation: GP surrogate model balances Exploration vs Exploitation
gp = GaussianProcessRegressor(kernel=Matern(nu=2.5))
gp.fit(X_sampled, y_sampled)
mu, sigma = gp.predict(X_candidates, return_std=True)
ucb_acquisition = mu + 2.0 * sigma  # Pick argmax(ucb) as next query point"""),

        _model("7. Learning Curves & Validation Curves", "Model Diagnostics", """
from sklearn.model_selection import learning_curve, validation_curve

# Learning curve across sample size N (diagnoses high bias vs high variance vs data scarcity)
train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, train_sizes=[0.1, 0.3, 0.5, 0.8, 1.0], cv=5
)

# Validation curve across a single hyperparameter (pinpoints under/over-fitting sweet spot)
vc_train, vc_val = validation_curve(
    estimator, X, y, param_name="max_depth", param_range=range(1, 20), cv=5
)"""),

        _model("8. Significance Testing (McNemar & Paired Bootstrap)", "Statistical Tests", """
import numpy as np
import scipy.stats as stats

# McNemar's test for paired classification:
# b: Model A right & Model B wrong; c: Model A wrong & Model B right
chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

# Paired Bootstrap for regression models:
boot_diffs = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
    mae_diff = np.mean(np.abs(y_test[idx] - pred_a[idx])) - np.mean(np.abs(y_test[idx] - pred_b[idx]))
    boot_diffs.append(mae_diff)
ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])"""),
    ],
}