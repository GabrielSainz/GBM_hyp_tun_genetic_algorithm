def prediction_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        
        # Initialize the tree with a root node
        self.root = Node(X, y)
        
        # Fit the tree recursively
        self._fit_node(self.root, depth=0)
        
    def _fit_node(self, node, depth):
        # Check if the node is a leaf or if the maximum depth has been reached
        if node.is_leaf or (self.max_depth is not None and depth >= self.max_depth):
            return
        
        # Find the best split for the current node
        best_split = self._find_best_split(node.X, node.y)
        
        if best_split is not None:
            # Split the node into two children and fit them recursively
            left_X, right_X, left_y, right_y = best_split
            node.left = Node(left_X, left_y)
            node.right = Node(right_X, right_y)
            self._fit_node(node.left, depth+1)
            self._fit_node(node.right, depth+1)
    
    def _find_best_split(self, X, y):
        # Iterate over all features and thresholds to find the best split
        best_split = None
        min_error = float('inf')
        
        for feature in range(self.n_features):
            for threshold in X[:, feature]:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                
                # Calculate the error of the split
                error = prediction_error(left_y, np.mean(left_y)) + prediction_error(right_y, np.mean(right_y))
                
                # Update the best split if necessary
                if error < min_error:
                    min_error = error
                    best_split = (X[left_mask], X[right_mask], left_y, right_y)
        
        return best_split
    
    def predict(self, X):
        # Initialize an array to store the predictions
        y_pred = np.empty(X.shape[0])
        
