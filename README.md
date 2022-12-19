# fed-baye
Bayesian FedRec with Differential Privacy

# Team Members
Kevin Schattin (ks3936) 
Lynn Kim (ck3127)

# description of the project
A common approach to ensuring privacy during the training of federated recommendation systems is the use of differential privacy. Differential privacy aims to solve the problem of gradient leakage, in which private user data can be recovered by third-party attackers during the transmission of gradients from local models on user edge devices to a central server. Differential privacy works by modifying the local model gradients with noise before transmitting them to the central server, thereby disguising the gradients themselves, as well as the underlying data on which they are based. One particular pitfall of differential privacy is the fact that it generally leads to poorer model performance since the noisy gradients can produce weight updates that are not optimal.

Our goal is to eliminate the tradeoff between user privacy and model performance through the use of Bayesian Neural Networks. Our belief is that the inherent uncertainty in the parameters of a Bayesian Neural Network will serve to mitigate the unintended effects of model updates that are based on noisy gradients produced via differential privacy. We believe that a recommender system built using both Bayesian Neural Networks and differential privacy will not only be secure, but will also be accurate.

# description of the repository

# example commands to execute the code   

# Results 
