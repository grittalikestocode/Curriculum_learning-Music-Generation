# KL Divergence Evaluation Script for MIDI Feature Distributions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, ttest_ind
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity
import pretty_midi
from midi_analyzer import MidiAnalyzer  # Replace with actual import if needed

# --- Bandwidth estimator ---
def Scott_bandwidth(X):
    n = len(X)
    d = 1
    return np.power(n, -1. / (d + 4)) * np.std(X)

# --- KDE function ---
def kernel_density_estimation(X, N, lower_r, upper_r):
    X = X.reshape(-1, 1)
    X_plot = np.linspace(lower_r, upper_r, N)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=Scott_bandwidth(X)).fit(X)
    log_density = kde.score_samples(X_plot)
    pdf = np.exp(log_density)
    return [pdf, X_plot]

# --- KL divergence for two PDFs ---
def get_KL_divergence(sample1, sample2, N, plot=False):
    min1, min2 = min(sample1), min(sample2)
    max1, max2 = max(sample1), max(sample2)
    lower = min(min1, min2)
    upper = max(max1, max2)
    tot_range = abs(upper - lower)
    lower_bound = lower - tot_range * 0.2
    upper_bound = upper + tot_range * 0.2
    kde1 = kernel_density_estimation(sample1, N, lower_bound, upper_bound)
    kde2 = kernel_density_estimation(sample2, N, lower_bound, upper_bound)
    kl = entropy(kde1[0], kde2[0])
    if plot:
        plt.plot(kde1[1], kde1[0], label="Intra")
        plt.plot(kde2[1], kde2[0], label="Inter")
        plt.legend()
        plt.show()
    return kl

# --- KL per feature ---
def get_KL_divergence_per_feature(h1, h2, plot=False):
    KL_per_feature = []
    for i in range(len(h1)):
        min_len = min(len(h1[i]), len(h2[i])) - 1
        KL = get_KL_divergence(h1[i][:min_len], h2[i][:min_len], 1000, plot)
        KL_per_feature.append(KL)
    return KL_per_feature

# --- Distance histogram matrix ---
def get_distances_hist(matrix1, matrix2):
    n_features = matrix1.shape[1]
    histogram_matrix = []
    for i in range(n_features):
        row1 = matrix1[:, i].reshape(-1, 1)
        row2 = matrix2[:, i].reshape(-1, 1)
        distances_matrix = euclidean_distances(row1, row2)
        tri = np.triu(distances_matrix) if distances_matrix.shape[0] <= distances_matrix.shape[1] else np.tril(distances_matrix)
        distances = np.reshape(tri, -1)
        distances = distances[distances != 0]
        histogram_matrix.append(distances)
    return histogram_matrix

# --- Extract features from a folder of MIDI files (.mid or .midi) ---
def extract_feature_matrix(folder_path, analyzer, features):
    matrix = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.mid', '.midi')):
            try:
                midi_path = os.path.join(folder_path, file)
                # midi_data = pretty_midi.PrettyMIDI(midi_path)
                result = analyzer.analyze_midi_file(midi_path)
                matrix.append([result[feat] for feat in features])
            except Exception as e:
                print(f"Failed to process {file}: {e}")
    return np.array(matrix)

# --- Main evaluation ---
def evaluate_kl(real_folder, gen_folder, features,label):
    analyzer = MidiAnalyzer()
    real_matrix = extract_feature_matrix(real_folder, analyzer, features)
    gen_matrix = extract_feature_matrix(gen_folder, analyzer, features)
    inter = get_distances_hist(real_matrix, gen_matrix)
    intra = get_distances_hist(gen_matrix, gen_matrix)
    # return get_KL_divergence_per_feature(intra, inter, plot=True)
    scores = get_KL_divergence_per_feature(intra, inter, plot=False)
    return dict(zip(features, scores)), label

# --- Save results to CSV ---
def save_kl_results_to_csv(result_dicts, filename="kl_results.csv"):
    df = pd.DataFrame(result_dicts)
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")


# --- T-test comparison ---
def compare_features_ttest(matrix1, matrix2, feature_names):
    result = []
    for i, name in enumerate(feature_names):
        vals1 = matrix1[:, i]
        vals2 = matrix2[:, i]
        stat, pval = ttest_ind(vals1, vals2, equal_var=False)
        result.append({"feature": name, "t_stat": stat, "p_value": pval})
    return pd.DataFrame(result)

# --- Save t-test result ---
def save_ttest_results(real_dir, model_dirs, features, output_file="ttest_results.csv"):
    analyzer = MidiAnalyzer()
    real_matrix = extract_feature_matrix(real_dir, analyzer, features)
    results = []
    for label, folder in model_dirs.items():
        model_matrix = extract_feature_matrix(folder, analyzer, features)
        t_df = compare_features_ttest(real_matrix, model_matrix, features)
        t_df.insert(0, 'model', label)
        results.append(t_df)
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Saved t-test results to {output_file}")

    

features_to_compare = [ 'pitch_entropy','polyphony','note_density','qualified_note_ratio','pitch_consonance_score','information_rate','ssm','qualified_rythm_freq','rhythmic_intensity','avg_pitch_interval','avg_ioi','empty_bar_ratio'
]
# maestro = evaluate_kl("generated_music/maestro_short", "generated_music/maestro_short", features_to_compare)
# kl_60 = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl60", features_to_compare)
# kl_60_lr = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl60_lr", features_to_compare)
# kl_80 = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl80", features_to_compare)
# baseline = evaluate_kl("generated_music/maestro_short", "generated_music/baseline", features_to_compare)
# print("Maestro:", dict(zip(features_to_compare, maestro)))
# print("KL vs Gen60:", dict(zip(features_to_compare, kl_60)))
# print("KL vs Gen60_LR:", dict(zip(features_to_compare, kl_60_lr)))
# print("KL vs Gen80:", dict(zip(features_to_compare, kl_80)))
# print("Baseline", dict(zip(features_to_compare, baseline)))


maestro, _ = evaluate_kl("generated_music/maestro_short", "generated_music/maestro_short", features_to_compare, label="maestro")
kl_60, _ = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl60", features_to_compare, label="CL60")
kl_60_lr, _ = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl60_lr", features_to_compare, label="CL60_lr")
kl_80, _ = evaluate_kl("generated_music/maestro_short", "generated_music/gen_cl80", features_to_compare, label="CL80")
baseline, _ = evaluate_kl("generated_music/maestro_short", "generated_music/baseline", features_to_compare, label="Baseline")

all_results = [
    {"model": "maestro", **maestro},
    {"model": "CL60", **kl_60},
    {"model": "CL60_lr", **kl_60_lr},
    {"model": "CL80", **kl_80},
    {"model": "Baseline", **baseline},
]

save_kl_results_to_csv(all_results, "kl_summary.csv")


models = {
    "CL60": "generated_music/gen_cl60",
    "CL60_lr": "generated_music/gen_cl60_lr",
    "CL80": "generated_music/gen_cl80",
    "Baseline": "generated_music/baseline"
}

save_ttest_results("generated_music/maestro_short", models, features_to_compare, "ttest_results.csv")

