from sklearn import datasets

class IrisClassifier(SupervisedBaselines):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                    
    def load_data(self):
        X, y = datasets.load_iris(return_X_y=True)
        self.X = X[y != 0]
        self.y = y[y != 0].astype(float)

        hyperparam_configs = {'svm-classifier': get_param_spec('svm-classifier')}
        #hyperparam_configs = get_all_param_specs(all_regressors=True)

    def fit(self):
        super().fit(self.X, self.y)

iris_model = IrisClassifier()
iris_model.fit()
