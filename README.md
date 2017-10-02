# Machine-learning-toolkits-with-python
Practical machine learning toolkits with Python 

## Ensemble methods (Bagging classifiers vs. Voting classifiers)

- Bagging & voting classifiers with Scikit-learn
- "Ensembles are well established as a method for obtaining highly accurate classiers by combining less accurate ones." (Dietterich 2000)
- Related papers
 1) Dietterich, T. G. (2000). Ensemble methods in machine learning. Multiple classifier systems, 1857, 1-15.
 2) Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.

![alt text](http://www.datakit.cn/images/machinelearning/EnsembleLearning_Combining_classifiers.jpg)

## Evaluation metrics for unbalanced data (ROC vs Informedness)

- ROC curve & Bookmakers' informedness/markedness with Scikit-learn
- "ROC graphs are a very useful tool for visualizing and evaluating classifiers. They are able to provide a richer measure of classification performance than scalar measures such as accuracy, error rate or error cost." (Fawcett 2006)
- "Note that while Informedness is a deep measure of how consistently the Predictor predicts the Outcome by combining surface measures about what proportion of Outcomes are correctly predicted, Markedness is a deep measure of how consistently the Outcome has the Predictor as a Marker by combining surface measures about what proportion of Predictions are correct. " (Powers 2011)
- Related papers
 1) Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition letters, 27(8), 861-874.
 2) Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation.
 
![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ROC_curves.svg/300px-ROC_curves.svg.png)

## Feature scaling (mean substraction vs normalization)

- Feature scaling with NumPy
- Related material: Lecture note from cs231n (lecture 6 - Training neural networks) 

![alt text](http://cs231n.github.io/assets/nn2/prepro1.jpeg)

## Hyperparmeter tuning (Grid search vs. Random search)

- Grid search & random search with Scikit-learn
- "Compared with neural networks configured by a pure grid search, we find that random search over the same domain is able to find models that are as good or better within a small fraction of the computation time." (Bergstra and Bengio 2012)
- Related paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb), 281-305.

![alt text](https://cdn-images-1.medium.com/max/1600/1*ZTlQm_WRcrNqL-nLnx6GJA.png)


## Model selection (Bootstrap vs. Cross-validation)

- Bootstrap & cross-validation with Scikit-learn
- "Our results indicate that for real-world datasets similar to ours, the best method to use for model selection is ten-fold stratified cross validation, even if computation power allows using more folds
- Related paper: Kohavi, R. (1995, August). A study of cross-validation and bootstrap for accuracy estimation and model selection. In Ijcai (Vol. 14, No. 2, pp. 1137-1145).

![alt_text](https://sebastianraschka.com/images/faq/evaluate-a-model/k-fold.png)


## Optimization methods (first-order vs second-order)

- Adam & L-bfgs optimization for neural networks
- "The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters" (Kingma and Ba 2015)
- "Our numerical tests indicate that the L-BFGS method is faster than the method of Buckley and LeNir, and is better able to use additional storage to accelerate convergence" (Liu and Nocedal 1989)

![alt_text](http://i.imgur.com/pD0hWu5.gif?1)
