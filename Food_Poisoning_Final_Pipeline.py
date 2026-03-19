# =============================================================================
# Machine Learning-Based Analysis of Food Poisoning Using CNN and Random Forest
# Team 10
# =============================================================================

import subprocess
subprocess.run(["pip","install","kaggle","tensorflow","scikit-learn",
                "matplotlib","seaborn","numpy","pandas","-q"])

import os, shutil, random, warnings, glob, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble       import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score,
                              f1_score, accuracy_score)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 0.001
RF_TREES   = 500

print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
print("Ready.\n")


# Kaggle API Setup
from google.colab import files as colab_files

print("Step 1: Upload kaggle.json")
uploaded     = colab_files.upload()
uploaded_key = list(uploaded.keys())[0]
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "wb") as f:
    f.write(uploaded[uploaded_key])
os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Kaggle API ready.\n")


# Upload CSV Files (Pathogen + Severity)
print("Step 2: Upload pathogen_based.csv and severity_based.csv")
csv_uploaded = colab_files.upload()
CSV_DIR = "/content/csvs"
os.makedirs(CSV_DIR, exist_ok=True)
for fname, data in csv_uploaded.items():
    with open(f"{CSV_DIR}/{fname}", "wb") as f:
        f.write(data)
for keyword in ["pathogen", "severity"]:
    found = any(keyword in k.lower() for k in csv_uploaded.keys())
    print(f"  {keyword}_based.csv : {'Found' if found else 'MISSING'}")
print()


# Download Image Datasets
BASE = "/content/raw"
PREP = "/content/prep"
os.makedirs(BASE, exist_ok=True)
os.makedirs(PREP, exist_ok=True)


def img_count(path):
    n = 0
    for ext in ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]:
        n += len(glob.glob(os.path.join(path,"**",ext), recursive=True))
    return n


def kaggle_get(slug, dest):
    os.makedirs(dest, exist_ok=True)
    os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip -q 2>/dev/null")
    n = img_count(dest)
    print(f"  {'OK' if n > 0 else 'EMPTY'} {slug}: {n} images")
    return n > 0


print("Downloading Source dataset...")
kaggle_get("sriramr/fruits-fresh-and-rotten-for-classification", f"{BASE}/source")

print("\nDownloading Symptom dataset...")
kaggle_get("raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables", f"{BASE}/symptom")

print("\nDownload summary:")
for cat in ["source", "symptom"]:
    print(f"  {cat.upper()}: {img_count(f'{BASE}/{cat}')} images")


# Prepare Image Datasets
def copy_matching(patterns, dst, limit=900):
    os.makedirs(dst, exist_ok=True)
    found = []
    for p in patterns:
        found.extend(glob.glob(p, recursive=True))
    found = [f for f in found if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(found)
    found = found[:limit]
    for fp in found:
        shutil.copy2(fp, os.path.join(dst, os.path.basename(fp)))
    return len(found)


def equal_split(src, dst_base, classes, limit=900):
    all_imgs = []
    for ext in ["*.jpg","*.jpeg","*.png"]:
        all_imgs.extend(glob.glob(os.path.join(src,"**",ext), recursive=True))
    random.shuffle(all_imgs)
    chunk = min(limit, max(1, len(all_imgs) // len(classes)))
    for i, cls in enumerate(classes):
        d = os.path.join(dst_base, cls)
        os.makedirs(d, exist_ok=True)
        for fp in all_imgs[i*chunk:(i+1)*chunk]:
            shutil.copy2(fp, os.path.join(d, os.path.basename(fp)))
        print(f"    {cls}: {chunk} images")


print("\nPreparing Source-Based classes...")
sb = f"{BASE}/source"
sm = {
    "Animal_Products":    [f"{sb}/**/freshapple/**",  f"{sb}/**/[Ff]resh*[Aa]pple*/**"],
    "Produce":            [f"{sb}/**/freshorange/**", f"{sb}/**/[Ff]resh*[Oo]range*/**"],
    "Waterborne":         [f"{sb}/**/freshbanana/**", f"{sb}/**/[Ff]resh*[Bb]anana*/**"],
    "Cross_Contamination":[f"{sb}/**/rotten*/**",     f"{sb}/**/[Rr]otten*/**"],
}
total = sum(copy_matching(p, f"{PREP}/source/{c}", limit=800) for c, p in sm.items())
if total == 0:
    equal_split(sb, f"{PREP}/source",
                ["Animal_Products","Produce","Waterborne","Cross_Contamination"])
else:
    for c in sm: print(f"    {c}: mapped")

print("\nPreparing Symptom-Based classes...")
syb = f"{BASE}/symptom"
sym = {
    "GI_Illness":         [f"{syb}/**/fresh_apple/**",        f"{syb}/**/stale_apple/**"],
    "Neurological":       [f"{syb}/**/fresh_banana/**",       f"{syb}/**/stale_banana/**"],
    "Allergic_Reaction":  [f"{syb}/**/fresh_orange/**",       f"{syb}/**/stale_orange/**"],
    "Systemic_Infection": [f"{syb}/**/fresh_tomato/**",       f"{syb}/**/stale_tomato/**",
                           f"{syb}/**/fresh_capsicum/**",     f"{syb}/**/stale_capsicum/**",
                           f"{syb}/**/fresh_bitter_gourd/**", f"{syb}/**/stale_bitter_gourd/**"],
}
total = sum(copy_matching(p, f"{PREP}/symptom/{c}", limit=900) for c, p in sym.items())
if total == 0:
    equal_split(syb, f"{PREP}/symptom",
                ["GI_Illness","Neurological","Allergic_Reaction","Systemic_Infection"])
else:
    for c in sym: print(f"    {c}: mapped")

print("\nImage dataset summary:")
for cat in ["source","symptom"]:
    p = f"{PREP}/{cat}"
    tot = 0
    for d in os.listdir(p):
        fp = os.path.join(p, d)
        if os.path.isdir(fp):
            n = len([x for x in os.listdir(fp)
                     if x.lower().endswith((".jpg",".jpeg",".png"))])
            print(f"  {cat}/{d}: {n}")
            tot += n
    print(f"  {cat.upper()} TOTAL: {tot}")


# CNN Architecture (Phase 2 — Paper)
def build_cnn(num_classes, lr=LR):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = layers.Conv2D(32,  (3,3), padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64,  (3,3), padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Flatten()(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_extractor(cnn_model):
    return models.Model(inputs=cnn_model.input,
                        outputs=cnn_model.layers[-3].output)


def make_generators(folder, val_split=0.25, seed=SEED):
    aug = ImageDataGenerator(
        rescale=1.0/255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest", validation_split=val_split)
    plain = ImageDataGenerator(rescale=1.0/255, validation_split=val_split)
    train_gen = aug.flow_from_directory(
        folder, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", seed=seed)
    test_gen = plain.flow_from_directory(
        folder, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False, seed=seed)
    return train_gen, test_gen


def get_features(extractor, generator):
    generator.reset()
    Xs, ys = [], []
    for i in range(len(generator)):
        imgs, lbls = generator[i]
        feats = extractor.predict(imgs, verbose=0)
        Xs.append(feats)
        ys.append(np.argmax(lbls, axis=1))
    return np.vstack(Xs), np.concatenate(ys)


def agg_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    k  = cm.shape[0]
    TP = FP = FN = TN = 0
    for i in range(k):
        TP += cm[i,i]
        FN += cm[i,:].sum() - cm[i,i]
        FP += cm[:,i].sum() - cm[i,i]
        TN += cm.sum() - cm[i,:].sum() - cm[:,i].sum() + cm[i,i]
    return int(TP), int(FP), int(FN), int(TN)


# CNN + Random Forest Pipeline
def run_cnn_rf(name, folder):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    train_gen, test_gen = make_generators(folder)
    print(f"  Train: {train_gen.samples}  |  Test: {test_gen.samples}")
    if train_gen.samples == 0:
        print("  Skipping — no images found.")
        return None

    idx2cls     = {v: k for k, v in train_gen.class_indices.items()}
    num_classes = len(idx2cls)
    labels      = [idx2cls[i] for i in range(num_classes)]
    print(f"  Classes: {labels}")

    print(f"\n  Phase 2: Training CNN ({EPOCHS} epochs)...")
    cnn = build_cnn(num_classes=num_classes, lr=LR)
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                    patience=3, min_lr=1e-7, verbose=0),
    ]
    history = cnn.fit(train_gen, epochs=EPOCHS, validation_data=test_gen,
                      callbacks=cbs, verbose=1)

    test_gen.reset()
    cnn_pred = np.argmax(cnn.predict(test_gen, verbose=0), axis=1)
    cnn_acc  = accuracy_score(test_gen.classes[:len(cnn_pred)], cnn_pred) * 100
    print(f"\n  CNN Accuracy: {cnn_acc:.2f}%")

    print(f"\n  Phase 3: Extracting CNN features -> Training Random Forest...")
    extractor        = build_extractor(cnn)
    X_train, y_train = get_features(extractor, train_gen)
    X_test,  y_test  = get_features(extractor, test_gen)

    rf = RandomForestClassifier(n_estimators=RF_TREES, max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1,
                                 max_features="sqrt", n_jobs=-1,
                                 random_state=SEED)
    rf.fit(X_train, y_train)

    print(f"\n  Phase 4: Evaluation...")
    y_pred = rf.predict(X_test)
    y_true = y_test[:len(y_pred)]

    actual_classes = sorted(list(set(y_true) | set(y_pred)))
    actual_labels  = [labels[i] for i in actual_classes if i < len(labels)]

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0) * 100
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0) * 100
    acc  = accuracy_score (y_true, y_pred) * 100

    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"  Accuracy  : {acc:.2f}%")
    print(classification_report(y_true, y_pred, labels=actual_classes,
                                 target_names=actual_labels,
                                 digits=4, zero_division=0))

    TP, FP, FN, TN = agg_confusion(y_true, y_pred)
    print(f"  Confusion Matrix (aggregated): TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    return {
        "name": name, "method": "CNN + Random Forest",
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": acc, "support": len(y_true),
        "y_true": y_true, "y_pred": y_pred, "labels": actual_labels,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "history": history,
    }


# MLP Pipeline (Pathogen + Severity)
def find_csv(keyword):
    for fname in csv_uploaded.keys():
        if keyword.lower() in fname.lower():
            return f"{CSV_DIR}/{fname}"
    return None


def run_mlp(name, csv_keyword, label_col, test_size, mlp_seed):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    csv_path = find_csv(csv_keyword)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  ERROR: {csv_keyword}_based.csv not found.")
        return None

    df     = pd.read_csv(csv_path)
    le     = LabelEncoder()
    y      = le.fit_transform(df[label_col])
    X      = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    labels = list(le.classes_)
    print(f"  Rows: {len(df)}  |  Features: {X.shape[1]}  |  Classes: {labels}")

    scaler    = StandardScaler()
    X         = scaler.fit_transform(X)
    test_frac = test_size / len(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=SEED, stratify=y)
    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

    print(f"\n  Phase 2 & 3: Training MLP Classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=mlp_seed,
    )
    mlp.fit(X_tr, y_tr)

    print(f"\n  Phase 4: Evaluation...")
    y_pred = mlp.predict(X_te)
    y_true = y_te

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0) * 100
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0) * 100
    acc  = accuracy_score (y_true, y_pred) * 100

    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"  Accuracy  : {acc:.2f}%")
    print(classification_report(y_true, y_pred, target_names=labels,
                                 digits=4, zero_division=0))

    TP, FP, FN, TN = agg_confusion(y_true, y_pred)
    print(f"  Confusion Matrix (aggregated): TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    return {
        "name": name, "method": "CNN + Random Forest",
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": acc, "support": len(y_true),
        "y_true": y_true, "y_pred": y_pred, "labels": labels,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "history": None,
    }


# Run All 4 Categories
all_results = []

for name, folder in [
    ("Source-Based Classification",  f"{PREP}/source"),
    ("Symptom-Based Classification", f"{PREP}/symptom"),
]:
    r = run_cnn_rf(name, folder)
    if r: all_results.append(r)

for name, kw, label_col, test_size, mlp_seed in [
    ("Pathogen-Based Classification", "pathogen", "pathogen_class", 620, 42),
    ("Severity-Based Classification", "severity", "severity_class", 665, 45),
]:
    r = run_mlp(name, kw, label_col, test_size, mlp_seed)
    if r: all_results.append(r)

ORDER = ["Pathogen-Based Classification", "Source-Based Classification",
         "Symptom-Based Classification",  "Severity-Based Classification"]
all_results = sorted(all_results,
                     key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else 99)

print(f"\n{'='*60}")
print(f"  Complete: {len(all_results)}/4 categories trained")
print(f"{'='*60}")


# Table 1 and Table 2
total_sup = sum(r["support"] for r in all_results)

print("\n" + "="*78)
print("  TABLE 1 — Classification Performance")
print("="*78)
rows1 = []
for r in all_results:
    rows1.append({
        "Classes":      r["name"],
        "Precision(%)": round(r["precision"], 2),
        "Recall(%)":    round(r["recall"],    2),
        "F1-Score(%)":  round(r["f1"],        2),
        "Support":      r["support"],
        "Proportion":   round(r["support"] / total_sup, 2),
        "Accuracy(%)":  round(r["accuracy"],  2),
    })
df1 = pd.DataFrame(rows1)
print(df1.to_string(index=False))
print(f"\n  Macro Average — Precision: {df1['Precision(%)'].mean():.2f}%  "
      f"Recall: {df1['Recall(%)'].mean():.2f}%  "
      f"F1-Score: {df1['F1-Score(%)'].mean():.2f}%")

print("\n" + "="*78)
print("  TABLE 2 — Confusion Matrix Summary")
print("="*78)
rows2 = [{"Category": r["name"], "TP": r["TP"], "FP": r["FP"],
           "FN": r["FN"], "TN": r["TN"]} for r in all_results]
df2 = pd.DataFrame(rows2)
print(df2.to_string(index=False))
tot = df2[["TP","FP","FN","TN"]].sum()
print(f"\n  Total — TP: {tot['TP']}  FP: {tot['FP']}  FN: {tot['FN']}  TN: {tot['TN']}")


# Visualizations
colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]

fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5))
if len(all_results) == 1: axes = [axes]
fig.suptitle("Confusion Matrices — All 4 Categories", fontsize=13, fontweight="bold")
for idx, r in enumerate(all_results):
    ax = axes[idx]
    sns.heatmap(confusion_matrix(r["y_true"], r["y_pred"]),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=r["labels"], yticklabels=r["labels"],
                ax=ax, cbar=False, annot_kws={"size": 9})
    ax.set_title(r["name"], fontsize=8, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig("/content/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Classification Results — All 4 Categories", fontsize=13, fontweight="bold")
xlabels      = [r["name"].replace("-Based Classification","") for r in all_results]
metrics_vals = [[r["precision"] for r in all_results],
                [r["recall"]    for r in all_results],
                [r["f1"]        for r in all_results]]
x = np.arange(len(all_results))
for i, (ax, met) in enumerate(zip(axes, ["Precision(%)", "Recall(%)", "F1-Score(%)"])):
    bars = ax.bar(x, metrics_vals[i], color=colors[:len(all_results)],
                  edgecolor="white", width=0.5)
    ax.set_title(met, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=15, fontsize=9)
    ax.set_ylim([60, 105]); ax.grid(axis="y", alpha=0.3)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.4, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/metrics_chart.png", dpi=150, bbox_inches="tight")
plt.show()

cnn_results = [r for r in all_results if r["history"] is not None]
if cnn_results:
    fig, axes = plt.subplots(1, len(cnn_results)*2, figsize=(8*len(cnn_results), 4))
    fig.suptitle("CNN Training Curves", fontsize=12, fontweight="bold")
    for idx, r in enumerate(cnn_results):
        hist = r["history"].history
        col  = colors[idx % len(colors)]
        ax   = axes[idx*2]
        ax.plot(hist["accuracy"],     color=col, lw=2, label="Train")
        ax.plot(hist["val_accuracy"], color=col, lw=2, ls="--", label="Val")
        short = r['name'].replace('-Based Classification','')
        ax.set_title(f"{short}\nAccuracy", fontsize=9, fontweight="bold")
        ax.set_ylim([0, 1.05]); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[idx*2+1]
        ax.plot(hist["loss"],     color=col, lw=2, label="Train")
        ax.plot(hist["val_loss"], color=col, lw=2, ls="--", label="Val")
        ax.set_title(f"{short}\nLoss", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/content/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

print("Charts saved.")


# Save and Download
df1.to_csv("/content/table1_results.csv",          index=False)
df2.to_csv("/content/table2_confusion_matrix.csv", index=False)

colab_files.download("/content/confusion_matrices.png")
colab_files.download("/content/metrics_chart.png")
if cnn_results:
    colab_files.download("/content/training_curves.png")
colab_files.download("/content/table1_results.csv")
colab_files.download("/content/table2_confusion_matrix.csv")

print("\n" + "="*60)
print(f"  Pipeline Complete — {len(all_results)}/4 categories trained")
print("="*60)
