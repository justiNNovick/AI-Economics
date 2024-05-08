A GPU is necessary to run these experiments. A singular V100 can be used in an EC2 instance, as this has compatible architecture.
With the above configuration, a single trial (200000 training episodes) will take over a day. Save at least 70GB of free storage for each trial.

It is necessary to install a downgraded Cuda driver (e.g., 1.13) that connects with the hardware used.

Python 3.8 was used to run the necessary experiments. 
In addition to the dependencies listed in requirements.txt, you must download torch in a compatible manner with the older cuda driver.
For instance, use this command.

pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

After running experiments, you can analyze data with the provided plotting scripts. 
plot_best_response.py is used to plot bar graphs for equilibrium analysis (scaled down as mentioned in the research paper), while plot_data.py 
is used to plot graphs for everything else in the environment, such as rewards, wages, prices, etc...
Each script takes an optional argument: the directory of the saved data (defaulted to the current directory of the script if no input is given)

To change utility functions before experiments, go to real_business_cycle -> rbc -> cuda -> firm_rbc
Next, open up firm_rbc.cu and regard lines 213-218
Follow the comments and adjust the script accordingly

For more specifics on running experiments, look at the README inside the real_business_cycle folder.
