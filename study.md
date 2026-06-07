# Battery Health (SOH) Prediction - EFREI Machine Learning Project
## Authors: Benjamin Arbousset, Loïc Cupif, Aurélien Ridard, DE INGE-2 2027

### Inspirations
- [Optimizing Neural Networks for SoH Prediction in Li-Ion Batteries: CNN, GRU, and GRU+Attention with Optuna](https://www.semanticscholar.org/paper/Optimizing-Neural-Networks-for-SoH-Prediction-in-Coronado-Zegarra/a39731b52d608a311c29cd8460fc78faf332513c)
    - Main inspiration for baseline
- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
- [CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719)
    - Tested in notebook 4 to compensate for domain shift between CALCE and NASA datasets - bad results
- [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)
    - Tested in nb05 to replace CORAL/DANN since GroupDRO doesn't need target data in batches (which is absent in LODO)

### Phase 1
The initial phase used only the NASA controlled and randomized datasets. We tested a random forest baseline built using hand crafted scalars against a CNN, LSTM, and a CNN-LSTM hybrid. However, it was impossible to beat RF at this stage. However, we discovered that to match RF, we had to use voltage indexed curves (and time as a feature). Performance on RW was acceptable.

### Phase 2
Since the volume of data was too low, we decided to add the CALCE CS2 and CX2 battery datasets. This made us go from 800 samples to about 13k. This allowed the LSTM and GRU implementations to beat RF. However, the hardest task is now dataset transfer, due to domain shift. The only model that survives Leave One Dataset Out (LODO) cross validation is BiGRU/BiLSTM. Adding CNN, attention, or using a Transformer only reduced the performance there. Though they did improve the performance on CALCE (the majority group).