import argparse

import seaborn as sns
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Data.")
    parser.add_argument("--save", required=True, type=str, help="Filename where to save result plot.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.data, "r") as f:
        lines = f.readlines()

    thresholds = []
    hs = []
    cs = []
    vs = []

    for line in lines:
        line = line.strip()
        threshold, h, c, v = line.split()

        thresholds.append(float(threshold))
        hs.append(float(h))
        cs.append(float(c))
        vs.append(float(v))

    # Plotting the data
    sns.set_theme()  # Set the style of the plot
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting the lines
    # sns.lineplot(x=thresholds, y=cs, label='cs')
    # sns.lineplot(x=thresholds, y=hs, label='hs')
    sns.lineplot(x=thresholds, y=vs, label='v', legend=None)

    peak_threshold = thresholds[vs.index(max(vs))]
    plt.text(peak_threshold+0.15, max(vs)-0.25, f'{max(vs)}', ha='left', va='bottom')
    plt.text(peak_threshold+0.15, 56.5, f'{peak_threshold}', ha='left', va='bottom')
    plt.scatter(peak_threshold, max(vs), color='red', zorder=5)  # Add a point at the peak
    plt.axvline(x=peak_threshold, color='red', linestyle='--')  # Add a vertical line at the peak

    # Adding labels and title
    plt.xlabel('Threshold coefficient')
    plt.ylabel('V-Measure')
    plt.title('V-Measure based on threshold coefficient')

    plt.savefig(args.save)


if __name__ == "__main__":
    main()