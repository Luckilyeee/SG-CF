# SG-CF: Shapelet-Guided Counterfactual Explanation for Time Series Data

# Abstract
EXplainable Artificial Intelligence (XAI) methods have gained much momentum lately given their ability to shed light on the decision function of opaque machine learning models. There are two dominating XAI paradigms: feature attribution and counterfactual explanation methods. While the first family of methods explains $why$ the model made a decision, counterfactual methods aim at answering \textit{what-if} the input is slightly different and results in another classification decision. Most research efforts have focused on answering the $why$ question for time series data modality. In this paper, we aim to answer the \textit{what-if} question by finding a good balance between a set of desirable counterfactual explanation properties. We propose Shapelet-guided Counterfactual Explanation  (SG-CF). This novel optimization-based model generates interpretable, intuitive post-hoc counterfactual explanations of time series classification models that balance validity, sparsity, proximity, interpretability, and contiguity. Our experimental results on nine real-world time-series datasets show that our proposed method can generate counterfactual explanations that balance all the desirable counterfactual properties compared to other competing baselines.

# Instructions
All python packages needed are listed in [pip-requirements.txt](pip-requirements.txt) file and can be installed simply using the pip command.
Run the SG_CF.ipynb file to get the evaluation results presented in the paper.
