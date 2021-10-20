
## Light Gradient Boosted Machine (LightGBM) Ensemble

- an efficient and effective implementation of the gradient boosting algorithm.
- extends the gradient boosting algorithm by adding a type of automatic feature selection as well as focusing on boosting examples with larger gradients. This can result in a dramatic speedup of training and improved predictive performance.
- GOSS, EFB
- Y=Base_tree(X)-lr*Tree1(X)-lr*Tree2(X)-lr*Tree3(X), where lr = learning rate
- Categorical and missing values support


```python
X, y = tps_march.drop("target", axis=1), tps_march[["target"]].values.flatten()

# Extract categoricals and their indices
cat_features = X.select_dtypes(exclude=np.number).columns.to_list()
cat_idx = [X.columns.get_loc(col) for col in cat_features]

# Convert cat_features to pd.Categorical dtype
for col in cat_features:
    X[col] = pd.Categorical(X[col])

# Unencoded train/test sets
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)
```
To specify the categorical features, pass a list of their indices to categorical_feature parameter in the fit method:

```python
# Model initialization is the same
eval_set = [(X_eval, y_eval)]

_ = lgbm_clf.fit(
    X_train,
    y_train,
    categorical_feature=cat_idx,  # Specify the categoricals
    eval_set=eval_set,
    early_stopping_rounds=150,
    eval_metric="logloss",
    verbose=False,
)

preds = lgbm_clf.predict_proba(X_eval)
loss = log_loss(y_eval, preds)
print(f"LGBM logloss with default cateogircal feature handling: {loss:.5f}")

```
### Gradient Bootsting
- Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems.
- Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting.
- Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

1. How Gradient Boosting Works?
<https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/>  
<https://bradleyboehmke.github.io/HOML/gbm.html>

A sequential ensemble approach

 each new tree in the sequence will focus on the training rows where the previous tree had the largest prediction errors.

## Packages
### Scikit learn
- the method via the scikit-learn wrapper classes: LGBMRegressor and LGBMClassifier. 
- Randomness is used in the construction of the model. This means that each time the algorithm is run on the same data, it will produce a slightly different model.

#### LightGBM Hyperparameters
the number of trees and tree depth, the learning rate, and the boosting type.

**Number of Trees**
- Recall that decision trees are added to the model sequentially in an effort to correct and improve upon the predictions made by prior trees. As such, more trees are often better.
- The number of trees can be set via the “n_estimators” argument and defaults to 100.

```python
# get a list of models to evaluate. Test different number of trees
def get_models():
	models = dict()
	trees = [10, 50, 100, 500, 1000, 5000]
	for n in trees:
		models[str(n)] = LGBMClassifier(n_estimators=n)
	return models
```

**Depth of Trees**
- Tree depth is controlled via the “max_depth” argument and defaults to an unspecified value as the default mechanism for controlling how complex trees are is to use the number of leaf nodes.
- There are two main ways to control tree complexity: the max depth of the trees and the maximum number of terminal nodes (leaves) in the tree. 

### Sklearn

## Quation
<https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/>

YouTube:
<Gradient Boost Part 2 (of 4): Regression Details>