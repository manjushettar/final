# CMPE 252 - Final
Music Recommendation Service

If you have the zip file:
1. Create a python environment (python3 -m env venv) and source into it
2. pip3 install -r requirements.txt
3. Run the script.py file in src folder
4. results/ folder contains your results

This the implementation of the Music Recommendation Service application. Currently, it is just a collection of scripts that simulate a bunch of agents that interact with song data.

This can easily be extended to real user interaction, as most of the required functionality for content-based and collaborative filtering can be found in dataloader.py

Running src/script.py enables the onboarding and recommender process, generating a simulation of recommendations being passed to each agent. The results are dumped into a results/ folder.

The data engineering and analysis part of this project is located in /data_eng
The 30,000 songs were light enough to dump into a csv file in /data
