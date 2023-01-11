#To optimize the parameters of GBM, we first need to define a fitness function that measures the performance of the model on a given set of parameters.
def fitness_function(params):
    n_estimators, learning_rate, max_depth = params
    model = GradientBoostingModel(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return -mse  # We need to return a negative value because the GA maximizes the fitness function
#It takes the model parameters as input and returns a score that reflects the model's performance.
