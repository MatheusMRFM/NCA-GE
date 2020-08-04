# NCA-GE Execution Guidelines

This repository implements the Network Centrality Approximation using Graph Embedding (NCA-GE), proposed in "Approximating Network Centrality Measures Using Node Embedding and Machine Learning". 

There are three folders:

1. **Graphs**: Contains the graphs used for training and testing. The training and testing graph files are not in this repo. You will have to generate them by running "NCA-GE/Graph_Generator.py". To do so, remember to fill the variables defined at the start of the file. By running this script, the graph files for the trianing and test will be generated and saved in "Graphs/Train/" and "Graphs/Test/", respectively (considering that the subfolder variables in the Graph_Generator.py are not altered). The current repo already has the files for the real networks used in the paper;
2. **NCA-GE**: Contains the code for generating the training and testing files (**Graph_Generator.py**), as well as the main code for the NCA-GE (**main.py**). To run the NCA-GE model, simply run the **main.py** file. Remember to first train the model (after generating the train and test files). The **Constants.py** stores all variables used during training. Remember to first set all variables in the latter file;
3. **Baseline**: Contains the code for the baseline model used in the paper. To configure this model, set the appropriate variables in **Constants.py**.
