# ML Algorithms from scratch

This repository offers three classic machine-learning implementations in pure Python/NumPy, no black-box libraries. Including Logistic Regression (SGD), Linear Regression (gradient descent), and ID3 Decision Tree implemented from scratch to break down core concepts through interactive, hands-on exploration. So we can see exactly how each algorithm learns from data.   

## Contents   

`data/`  
- `credit-risk-data.txt` - Test data for ID3-Decision-tree algorithm.

`plots/`  
- Loss plots of VLR experiments.  

`python/`  
- Three aforementioned algorithms in python. Each explained along with it's results below.  

## Vectorized Linear Regression  
Uses NumPy vector operations to perform gradient-descent linear regression. Compare its execution time and clarity against a basic loop-based approach. Loss plots from the experiments are attached below: 

<h4>Constant Learning Rate = 0.01, Varying Epochs</h4>
<img src="plots/Loss-Plot-1.png" alt="Loss Plot 1" width="600"/>

<h4>Constant Epochs = 100, Varying Learning Rate</h4>
<img src="plots/Loss-Plot-2.png" alt="Loss Plot 2" width="600"/>

## Stochastic-GD Logistic Regression    
Implements logistic regression training with stochastic gradient descent. You can adjust batch size and learning rate to observe their impact on convergence speed and the shape of the decision boundary.  


## ID3 Decision Tree  
Builds a decision tree on categorical data by computing entropy and information gain. Shows you how recursive splits form the tree and how leaf nodes decide the final class by majority vote.
ID3 DT Algorithm was tested on credit-risk-data.txt; Information Gains calculated by Attribute at different Node Levels are shown below:     

| **Attribute**      | **Root Node (IG)** | **2nd Node (IG)** | **3rd Node (IG)** | **4th Node (IG)** |
| ------------------ | ------------------ | ----------------- | ----------------- | ----------------- |
| **Income**         | **0.379**          | 0.000             | 0.000             | 0.000             |
| **Owns\_Property** | 0.252              | 0.000             | 0.000             | 0.000             |
| **Debt**           | 0.029              | 0.171             | **0.252**         | 0.000             |
| **Married?**       | 0.018              | **0.420**         | 0.000             | 0.000             |
| **Gender**         | 0.000              | 0.020             | **0.252**         | **1.000**         |

Final Decision Tree Structure for Credit-Risk Data:    

```
Income
├── High: Low
├── Medium: Low
└── Low
    └── Married?
        ├── Yes: High
        └── No
            └── Debt
                ├── Low: Low
                └── Medium
                    └── Gender
                        ├── Male: High
                        └── Female: Low
```

This repo provides a transparent, framework-free implementations of essential ML models.  
