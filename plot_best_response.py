#This script will plot the middle and end of the best response output (assuming 2 outputs were requested when best response was called)

import re
import os
import argparse
import matplotlib.pyplot as plt


def parse_br_files(directory):
    data = {}
    regex_pattern = r"rollout\s+[\w\d]+:\s*before BR training ([-\d\.]+) \((\d+\.\d+)\), after BR training ([-\d\.]+) \((\d+\.\d+)\), mean improvement ([-\d\.]+) \((\d+\.\d+)"


    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                print(file)
                with open(os.path.join(root, file), "r") as f:
                    iters = 0
                    for line in f:
                        match = re.search(regex_pattern, line)
                        print(match)
                        if match:
                            if "consumer" in file:
                                if "consumer middle" not in data:
                                    data["consumer middle"] = (float(match.group(1)), float(match.group(3)))
                                else:
                                    data["consumer end"] = (float(match.group(1)), float(match.group(3)))
                            elif "firm" in file:
                                if "firm middle" not in data:
                                    data["firm middle"] = (float(match.group(1)) / 10**4, float(match.group(3)) / 10**4)
                                else:
                                    data["firm end"] = (float(match.group(1)) / 10**4, float(match.group(3)) / 10**4)
                            elif "government" in file:
                                if "government middle" not in data:
                                    data["government middle"] = (float(match.group(1)) / 10**3, float(match.group(3)) / 10**3)
                                else:
                                    data["government end"] = (float(match.group(1)) / 10**3, float(match.group(3)) / 10**3)
                        iters += 1
                        if iters == 2:
                            break
    return data




def plot_bars(data):
    labels = list(data.keys())
    left_values = [value[0] for value in data.values()]
    right_values = [value[1] for value in data.values()]

    x = range(len(labels)) 
    width = 0.35  

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, left_values, width, label='Left Value', color='lightblue')
    rects2 = ax.bar([p + width for p in x], right_values, width, label='Right Value', color='plum')

    ax.set_ylabel('Mean Rewards')
    ax.set_title('Improvements in Mean Rewards with Best Response Training')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot values from best response output files.')
    parser.add_argument('directory_name', type=str, help='Directory containing the output files.', default=os.getcwd(), nargs='?')

    args = parser.parse_args()

    data = parse_br_files(args.directory_name)
    print(data)
    plot_bars(data)

if __name__ == "__main__":
    main()


                    