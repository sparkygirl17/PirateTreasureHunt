# Pirate Treasure Hunt
AI search algorithm game for AI 801 at Penn State University 

The AI agent, or pirate, is tasked with finding the treasure on the map and will come across various items that change its search method from the starting Breadth-First-Search (BFS) algorithm. There are two items to collect, and these items dynamically change the algorithm. The coin changes the search to the Greedy Best-First-Search, and the map changes to the A* algorithm. This simulation includes both this adaptive strategy as well as benchmark cases of BFS, Uniform Cost Search (UCS), Greedy Best-First-Search, and A* individually with each map selection. Performance is assessed based on steps, path cost, expanded nodes, unique expanded nodes, and computation time. Our findings are used to educate and demonstrate real-time interaction of these algorithms and showcase the core strengths and weaknesses of each search strategy.

## Active File Backing Streamlit

The main file to access the code that is running streamlit is titled Pirate_Treasure_Hunt_Final.py. This path is linked with a Streamlit Dashboard. We have the original Jupyter Notebook file titled as Pirate_Treasure_Hunt_Updated_Prototype (1).ipynb. This has the original code unedited with Streamlit integrations. 

## Running the Dashboard

The Pirate Treasure Hunt can be run by accessing the link provided below. 

Pirate Treasure Hunt Dashboard: 
https://piratetreasurehuntfinalpy-kbleogycrxw8qzsp9mkrue.streamlit.app/

The user can input their name for a welcome message and then select the desired dropdown options. The user can choose from three different maps and four different search methods. BFS, UCS, Greedy, A* and Adaptive Search. 

Adaptive Search is defined as the search strategy that dynamically switches between algorithms when specific items are found. The agent begins with the standard Breadth-First-Search strategy and will change methods when it is met with a coin and a map. Eventually completion is found when the agent reaches the treasure. Each map include a starting node and location for the treasure. 

The dropdown allows the user to select benchmark cases of each algorithm on its own with each map. This allows the user to have access to benchmark metrics and visual node exploration to enhance understanding. 

The end of the dashboard include the chart for which all metrics are displayed. The user can explore the visual display of each map and search combination while having access to the metrics for all combinations at all times. These metrics offer steps, path cost, expanded nodes, unique expanded nodes, planning time, planning calls, algorithm switches, and if the search succeeded. 

## Authors

This code was written by Hanbin Lee and edited for Streamlit integration by Valerie Stapleton in conjunction with Famela Dauz for the Pennsylvania State University Foundations of Artificial Intelligence course in the Fall of 2026. 

## Cite this Program

Stapleton, V., Lee, H., & Dauz, F. (2026). Pirate Treasure Hunt Final (Version 2.0.0) [Source code]. GitHub. [github.com](https://github.com/sparkygirl17/PirateTreasureHunt)

## Sources 

Russell, S., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed.). Pearson. https://aima.cs.berkeley.edu/ 
