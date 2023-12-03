import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns


def plot_doc_len_distribution_chart(num_words_records):
    plt.figure(figsize=(10, 6))

    # Get the unique word counts
    unique_word_counts = np.unique(num_words_records)
    # about 3 to 4 unique values in a bin
    num_bins = math.ceil(len(unique_word_counts) / 3)

    sns.histplot(num_words_records, bins=num_bins, kde=True)
    plt.title('Document Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()


def plot_num_docs_per_category_chart(category_counts):
    top_categories = category_counts.nlargest(20)
    other_sum = category_counts[~category_counts.index.isin(top_categories.index)].sum()
    top_categories['other'] = other_sum

    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
    ax.set_title('Number of Documents per Category', fontsize=20)
    ax.set_xlabel('Number of Documents', fontsize=16)
    ax.set_ylabel('Category', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.tight_layout()
    plt.show()


def plot_smooth_curve(data, title):
    # Create a smoother curve using interpolation
    xnew = np.linspace(0, len(data) - 1, 300)
    spl = make_interp_spline(range(len(data)), data, k=3)
    smooth_data = spl(xnew)

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, smooth_data, label=title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(f'{title} Over Epochs')
    plt.legend()
    plt.show()


def plot_stat_curves(train_data, test_data, stat_name):
    # Create smoother curves using interpolation
    xnew = np.linspace(0, len(train_data) - 1, 300)  # 300 points for smoothness

    train_spl = make_interp_spline(range(len(train_data)), train_data, k=3)
    train_smooth = train_spl(xnew)

    test_spl = make_interp_spline(range(len(test_data)), test_data, k=3)
    test_smooth = test_spl(xnew)

    plt.figure(figsize=(10, 6))
    plt.plot(xnew, train_smooth, label=f'Train {stat_name}')
    plt.plot(xnew, test_smooth, label=f'Test {stat_name}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Testing {stat_name} Over Epochs')
    plt.legend()
    plt.show()


