import pandas as pd
import numpy as np


class TreeNode(object):


    def __init__(self, index, split_value, feature_type, criterion):

        self.index = index
        self.split_value = split_value
        self.feature_type = feature_type
        self.criterion = criterion

        self.left_child = None
        self.right_child = None

        # if the node is a leaf node, set the label to the category that has
        # appeared the most times
        self.label = None

    def set_left_child(self, node):

        self.left_child = node

    def set_right_child(self, node):

        self.right_child = node

    def split(self, X_tree):
        if self.feature_type == 'num':  # use '<=' to split numeric feature
            left_mask = X_tree.iloc[:, self.index] <= self.split_value
        else:  # use '==' to split string feature
            left_mask = X_tree.iloc[:, self.index] == self.split_value

        return left_mask.values

    def get_impurity(self, X_tree, y_tree):
        if len(X_tree) == 0:
            return 0
        left_mask = self.split(X_tree)
        y_left = y_tree[left_mask]
        y_right = y_tree[~left_mask]
        return getattr(self, self.criterion)(y_tree, y_left, y_right)

    @staticmethod
    def gini(y_tree, y_left, y_right):

        if len(y_tree) == 0:
            return 0
        tree_prob_parent = np.unique(y_tree, return_counts=True)[1] / len(y_tree)
        gini_parent = 1 - np.sum(np.square(tree_prob_parent))

        for y_child in [y_left, y_right]:
            if len(y_child) == 0:
                continue
            prob_child = np.unique(y_child, return_counts=True)[1] / len(y_child)
            gini_child = float(len(y_child))/float(len(y_tree)) * (1 - np.sum(np.square(prob_child)))
            gini_parent -= gini_child
        return gini_parent

    @staticmethod
    def entropy(y, y_left, y_right):
        raise NotImplementedError

    @staticmethod
    def error(y, y_left, y_right):
        raise NotImplementedError


class DecisionTreeClassifier(object):
    def __init__(self, criterion='gini', max_depth=None, d=None,
                 random_state=0):

        assert criterion in ['gini', 'entropy',
                             'error'], 'Expect criterion is one of "gini", ' \
                                       '"entropy" or "error", bug got %s' % criterion
        self.criterion = criterion
        self.max_depth = max_depth
        self.d = d
        self.random_state = random_state

    def fit(self, X, y):


        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.squeeze(y.values)
        self.label_dtype = y.dtype


        if len(y.shape) >= 2:
            raise Exception(
                'Expect y has 1d dimension, bug got y.shape=' + str(
                    y.shape))


        milestones = []
        feature_types = []
        for col in X.columns:
            unique_features = np.sort(X[col].unique())
            milestones.append(unique_features)
            if type(X[col].dtype) == 'object':  # str features
                feature_types.append('str')
            else:
                feature_types.append('num')
        milestones = np.asarray(milestones)
        feature_types = np.asarray(feature_types)
        self.tree = self._build_tree(X, y, milestones, feature_types, 0)

    def _build_tree(self, X, y, milestones, feature_types, depth=0):

        assert len(X) == len(y)
        rgen = np.random.RandomState(self.random_state)

        if len(X) == 0 or (
                self.max_depth is not None and depth > self.max_depth):
            return None

        if np.all(y[0] == y):  # the whole data have a same category.
            node = TreeNode(None,None,None,self.criterion)
            node.label = y[0]
            return node


        best_node = None
        best_impurity = 0
        feature_indices = np.arange(len(feature_types))
        sampled_feature_types = feature_types
        sampled_milestones = milestones
        if self.d is not None:
            assert self.d <= len(feature_indices), 'Expect d <= X.shape[1], ' \
                                                   'but got d = %d' % self.d

            feature_indices = rgen.choice(feature_indices, self.d,
                                          replace=False)
            sampled_feature_types = feature_types[feature_indices]
            sampled_milestones = milestones[feature_indices]

        for findex, ftype, frange in zip(feature_indices,
                                           sampled_feature_types,
                                           sampled_milestones):
            for fvalue in frange:
                node = TreeNode(findex, fvalue, ftype, self.criterion)
                impurity = node.get_impurity(X, y)
                if impurity > best_impurity:
                    best_node = node
                    best_impurity = impurity
        if best_node is not None:
            left_mask = best_node.split(X)
            right_mask = ~left_mask
            X_left, y_left = X.loc[left_mask], y[left_mask]
            left_child = self._build_tree(X_left, y_left, milestones,
                                          feature_types, depth + 1)
            X_right, y_right = X.loc[right_mask], y[right_mask]
            right_child = self._build_tree(X_right, y_right, milestones,
                                           feature_types, depth + 1)
            if left_child is None and right_child is None:
                # current node is a leaf node
                catorgeris, counts = np.unique(y, return_counts=True)
                best_node.label = catorgeris[np.argmax(counts)]
            else:
                # Either there are no child nodes or there are two child nodes
                assert left_child is not None and right_child is not None

                best_node.set_left_child(left_child)
                best_node.set_right_child(right_child)
        else:
            best_node = TreeNode(None,None,None,self.criterion)
            catorgeris, counts = np.unique(y, return_counts=True)
            best_node.label = catorgeris[np.argmax(counts)]

        return best_node

    def _recursive_predict(self, X, node):

        if len(X) == 0:
            return np.asarray([],dtype=self.label_dtype)

        if node.label is not None:
            return np.asarray(len(X) * [node.label],dtype=self.label_dtype)

        left_mask = node.split(X)
        left_predict = self._recursive_predict(X.loc[left_mask],
                                               node.left_child)
        right_predict = self._recursive_predict(X.loc[~left_mask],
                                                node.right_child)
        predict = np.empty(shape=len(X),dtype=self.label_dtype)
        predict[left_mask] = left_predict
        predict[~left_mask] = right_predict

        return predict

    def predict(self, X):
        '''
        Returns predicted categories of `X`
        Args:
            X: pandas.DataFrame of numpy.ndarray
                input data
        Returns:
            pred_y: predicted categories of `X`
        '''
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return self._recursive_predict(X, self.tree)



