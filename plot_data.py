import numpy as np
import os
import re
import matplotlib.pyplot as plt
import argparse

def load_and_process_data(directory_path, agent_type):
    """ Load and process data based on the agent type. """
    if agent_type == 'firm':
        return load_and_process_firm_data(directory_path)
    else:
        return load_and_process_rewards(directory_path, agent_type)


def load_and_process_rewards(directory_path, agent_type):
    file_pattern = re.compile(rf'episode_(\d+)_({agent_type})\.npz$')
    final_file_pattern = re.compile(rf'episode_final_({agent_type})\.npz$')
    episode_rewards = []
    final_file_path = None

    for filename in os.listdir(directory_path):
        if final_file_pattern.match(filename):
            final_file_path = os.path.join(directory_path, filename)
        elif file_pattern.match(filename):
            episode_number = int(file_pattern.match(filename).group(1))
            file_path = os.path.join(directory_path, filename)
            data = np.load(file_path)
            
            if 'rewards' in data:
                rewards = data['rewards']
                average_rewards_per_timestep = np.mean(rewards, axis=1)  
                episode_rewards.append((episode_number, average_rewards_per_timestep))
            else:
                print(f"The file {filename} does not contain rewards data.")

    if final_file_path:
        process_final_file(final_file_path, episode_rewards, file_pattern)

    return sorted(episode_rewards, key=lambda x: x[0])

def process_final_file(final_file_path, episode_data, agent_type, final_episode_number=200000):
    """ Process the final episode file separately and add its data to the episode_data list. """
    data = np.load(final_file_path)
    if 'rewards' in data:
        rewards = data['rewards']
        average_rewards_per_episode = np.mean(rewards, axis=(1, 2))
        file_average_reward = np.mean(average_rewards_per_episode)
        episode_data.append((final_episode_number, file_average_reward))
    elif 'actions' in data:
        action_indices = data['actions']
        action_array = data['action_array']
        tax_rates = action_array[action_indices[:, :, 0].astype(int)]
        corporate_tax_rates = np.mean(tax_rates[:, :, 0], axis=1)
        income_tax_rates = np.mean(tax_rates[:, :, 1], axis=1)
        episode_data.append((final_episode_number, corporate_tax_rates, income_tax_rates))
    else:
        print(f"The final episode file {final_file_path} does not contain expected data.")

def load_and_process_firm_data(directory_path):
    """ Specific data processing for firm agent type to handle prices and wages. """
    file_pattern = re.compile(r'episode_(\d+)_firm\.npz$')
    final_file_pattern = re.compile(r'episode_final_firm\.npz$')
    num_firms = 10  
    price_indices = slice(0, num_firms)
    wage_indices = slice(num_firms, 2 * num_firms)

    prices = []
    wages = []
    final_file_path = None

    for filename in os.listdir(directory_path):
        if final_file_pattern.match(filename):
            final_file_path = os.path.join(directory_path, filename)
        match = file_pattern.match(filename)
        if match:
            episode_number = int(match.group(1))
            file_path = os.path.join(directory_path, filename)
            data = np.load(file_path)

            if 'states' in data:
                states = data['states']
                prices.append((episode_number, np.mean(states[..., price_indices], axis=(0, 1, 2))))
                wages.append((episode_number, np.mean(states[..., wage_indices], axis=(0, 1, 2))))
            else:
                print(f"The file {filename} does not contain 'states' data.")

    if final_file_path:
        process_final_firm_file(final_file_path, prices, wages, num_firms, final_episode_number=200000)

    return sorted(prices, key=lambda x: x[0]), sorted(wages, key=lambda x: x[0])

def process_final_firm_file(final_file_path, prices, wages, num_firms, final_episode_number):
    """ Process the final firm file to extract prices and wages. """
    data = np.load(final_file_path)
    if 'states' in data:
        states = data['states']
        price_indices = slice(0, num_firms)
        wage_indices = slice(num_firms, 2 * num_firms)
        final_prices = np.mean(states[..., price_indices], axis=(0, 1, 2))
        final_wages = np.mean(states[..., wage_indices], axis=(0, 1, 2))
        prices.append((final_episode_number, final_prices))
        wages.append((final_episode_number, final_wages))
    else:
        print("The final episode file does not contain 'states' data.")



def load_and_process_actions(directory_path, agent_type):
    """ Load and process actions based on the agent type, converting indices to tax rates. """
    file_pattern = re.compile(rf'episode_(\d+)_({agent_type})\.npz$')
    final_file_pattern = re.compile(rf'episode_final_({agent_type})\.npz$')
    tax_data = []
    final_file_path = None

    for filename in os.listdir(directory_path):
        if final_file_pattern.match(filename):
            final_file_path = os.path.join(directory_path, filename)
        elif file_pattern.match(filename):
            episode_number = int(file_pattern.match(filename).group(1))
            file_path = os.path.join(directory_path, filename)
            data = np.load(file_path)

            if 'actions' in data:
                action_indices = data['actions']
                action_array = data['action_array']
                tax_rates = action_array[action_indices[:, :, 0].astype(int)] 
                corporate_tax_rates = np.mean(tax_rates[:, :, 0], axis=1) *100 
                income_tax_rates = np.mean(tax_rates[:, :, 1], axis=1) * 100  # Convert to percentage
                tax_data.append((episode_number, corporate_tax_rates, income_tax_rates))
            else:
                print(f"The file {filename} does not contain actions data.")

    # Process the final episode
    if final_file_path:
        data = np.load(final_file_path)
        if 'actions' in data:
            # The action integer is the index from the action_array that contains the corporate, income tax rate as a 2-element list
            action_indices = data['actions']
            action_array = data['action_array']  
            tax_rates = action_array[action_indices[:, :, 0].astype(int)]
            corporate_tax_rates = np.mean(tax_rates[:, :, 0], axis=1)
            income_tax_rates = np.mean(tax_rates[:, :, 1], axis=1)
            final_episode_number = 200000  #last episode
            tax_data.append((final_episode_number, corporate_tax_rates, income_tax_rates))
        else:
            print("The final episode file does not contain actions data.")

    return sorted(tax_data, key=lambda x: x[0])



def plot_tax_data(data, title, xlabel, ylabel):
    """ Plotting function for tax data with separate lines for corporate and income tax. """
    if len(data) > 0:
        plt.figure(figsize=(10, 6))
        for tax_index, tax_label in enumerate(['Corporate Tax', 'Income Tax'], start=1):
            episode_numbers = [x[0] for x in data]
            tax_values = [np.mean(x[tax_index]) for x in data]  # Average tax rates per episode
            plt.plot(episode_numbers, tax_values, label=f'Average {tax_label}')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data available to plot.")


def plot_data(data, title, xlabel, ylabel):
    """ Generic plotting function for data that plots mean, 25th, and 75th percentiles. """
    if len(data) > 0 and isinstance(data[0][1], np.ndarray):
        episode_numbers = []
        mean_values = []
        p25_values = []
        p75_values = []

        for episode_number, array in data:
            mean_values.append(np.mean(array))
            p25_values.append(np.percentile(array, 25))
            p75_values.append(np.percentile(array, 75))
            episode_numbers.append(episode_number)

        plt.figure(figsize=(10, 6))
        plt.plot(episode_numbers, mean_values, linestyle='-', color='b', label='Mean')
        if "government" not in title.lower():
            plt.fill_between(episode_numbers, p25_values, p75_values, color='gray', alpha=0.5, label='25th-75th Percentile')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data available to plot.")


def main(experiment_directory='.'):
    """ Main function to orchestrate data loading and plotting for multiple agent types. """

    parser = argparse.ArgumentParser(description='Process and plot simulation data.')
    parser.add_argument('--experiment_directory', type=str, default='.', help='Directory where the experiment data is stored')
    args = parser.parse_args()
    experiment_directory = args.experiment_directory
    
    #specidal handling for government tax rates
    government_taxes = load_and_process_actions(experiment_directory, 'government')
    plot_tax_data(government_taxes, 'Government Taxes Over Training Episodes', 'Training Episode', 'Tax Rate (%)')

    #rewards for each agent
    agent_types = ['consumer', 'firm', 'government'] 
    for agent_type in agent_types:
        episode_rewards = load_and_process_rewards(experiment_directory, agent_type)
        plot_data(episode_rewards, f'Average Reward Over Training Episodes ({agent_type.capitalize()})',
                  'Training Episode', 'Average Reward')

    # Special handling for 'firm' prices and wages
    prices, wages = load_and_process_firm_data(experiment_directory)
    plot_data(prices, 'Average Prices Over Training Episodes (Firm)', 'Training Episode', 'Average Price')
    plot_data(wages, 'Average Wages Over Training Episodes (Firm)', 'Training Episode', 'Average Wage')


if __name__ == '__main__':
    main()
