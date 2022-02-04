# Intelligent Ledger Construction
This project explores a lightweight DLT ledger construction algorithm designed with IoT in mind using the IOTA ledger protocol. This project was originally mirrored from the IOTA Tangle ledger simulator [here](https://github.com/iotaledger/iotavisualization) with an added python backend for computation. 

It is recommended to set up the project in an IDE like Intellij Idea to repeatedly run tests and simulations.

### Biased Thompson Sampling (TS) for Ledger Construction
The main 'brain' behind decision-making is a multi-arm bandit called Biased TS. Isolated bandit tests comparing the approach to others is found in ./isolated_bandit.

The actual ledger construction algorithm python simulation is in ./intelligent_tip with an ai.gym environment for the Tangle in ./tangle_env. Ledger construction uses Biased TS in combination with Conservative Q-improvement from [here](https://arxiv.org/abs/1907.01180).

Simulations for ledger construction can be run isolated with the Flask backend in ./pyserver or with a visualization frontend in ./react_client. Run python scripts from ./.

#### Performance Analysis
Power and resource consumption scripts comparing various algorithms can be found in ./power_analysis.





