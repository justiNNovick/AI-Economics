import matplotlib.pyplot as plt
import numpy as np
import re
import os
from typing import List, Dict






def parse_br_files(directory: str) -> Dict[str, List[float]]:
    data = {}

    regex_pattern = r"before BR training (\d+\.\d+) \(\d+\.\d+\), after BR training (\d+\.\d+) \(\d+\.\d+\), mean improvement ([\-\d\.]+) \(\d+\.\d"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('br') and file.endswith('.txt'):
                agent_type = 'consumer' if 'consumer' in file else 'firm' if 'firm' in file else 'government'
                if agent_type not in data:
                    data[agent_type] = []
                elif agent_type in data:
                    agent_type_number = 1
                    while agent_type in data:
                        agent_type_number += 1
                        agent_type = agent_type + str(agent_type_number)
                    data[agent_type] = []

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    #assume this line works
                    matches = re.findall(regex_pattern, content)
                    ###print("matches", matches)

                    for match in matches:
                        before_training, after_training, improvement = map(float, match)
                        if 'firm' in agent_type:
                            before_training /= 10**4
                            after_training /= 10**4
                        elif 'government' in agent_type:
                            before_training /= 10**3
                            after_training /= 10**3
                        data[agent_type].append((before_training, after_training))

    return data

data = parse_br_files(".")
print(data)

labels = list(data.keys())
before_values = [x[0][0] for x in data.values()]  # First element of each tuple
after_diffs = [x[0][1] - x[0][0] for x in data.values()]  # Difference to reach the after value from the before value

x = range(len(labels))  # Label locations

fig, ax = plt.subplots()

# Create bars
bars_before = ax.bar(x, before_values, color='lightblue', label='Before Training')
bars_after = ax.bar(x, after_diffs, bottom=before_values, color='plum', label='After Training')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Agent Types')
ax.set_ylabel('Values')
ax.set_title('Before and After Training Values by Agent Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()


'''

# Define a function to parse the text files and extract the required information
def parse_br_files(directory: str) -> Dict[str, List[float]]:
    data = {'consumer': [], 'firm': [], 'government': []}
    regex_pattern = r"before BR training (\d+\.\d+) \(\d+\.\d+\), after BR training (\d+\.\d+) \(\d+\.\d+\), mean improvement ([\-\d\.]+) \(\d+\.\d+\)"

    # Walk through the directory and find files that match the criteria
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('br') and file.endswith('.txt'):
                agent_type = 'consumer' if 'consumer' in file else 'firm' if 'firm' in file else 'government'
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    content = f.read()
                    matches = re.findall(regex_pattern, content)

                    # If the pattern is found, process the data
                    for match in matches:
                        before_training, after_training, improvement = map(float, match)
                        # Adjust scale for firm and government mean rewards
                        if agent_type == 'firm':
                            before_training /= 10**4
                            after_training /= 10**4
                        elif agent_type == 'government':
                            before_training /= 10**3
                            after_training /= 10**3
                        data[agent_type].append((before_training, after_training))

    return data

# Define a function to generate a plot using Matplotlib
def plot_data(data: Dict[str, List[float]]):
    fig, ax = plt.subplots()

    # Set position of bar on X axis
    bar_width = 0.25
    positions = np.array(range(len(data['consumer'])))

    for idx, (agent_type, values) in enumerate(data.items()):
        before_training = [val[0] for val in values]
        after_training = [val[1] for val in values]
        improvements = np.array(after_training) - np.array(before_training)
        
        # Make the plot
        ax.bar(positions + idx * bar_width, before_training, color='blue', width=bar_width, label=f'{agent_type.capitalize()} Original')
        ax.bar(positions + idx * bar_width, improvements, bottom=before_training, color='purple', width=bar_width, label=f'{agent_type.capitalize()} Improvement')

    # Add labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Mean Rewards')
    ax.set_title('Mean Rewards and Best-Response Reward Improvements')
    ax.set_xticks(positions + bar_width)
    ax.set_xticklabels(['Middle', 'End'])
    ax.legend()

    plt.show()

# Example usage:
# Since we cannot actually read files, let's mock the result of the parse_br_files function
mock_data = {
    'consumer': [(2, 2.5), (3, 3.2)],
    'firm': [(4, 4.8), (5, 5.5)],
    'government': [(6, 6.3), (7, 7.1)]
}

# Plot the data
plot_data(mock_data)
'''