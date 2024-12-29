import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    sorted_indices = np.argsort(feature_vector)
    target_sorted = target_vector[sorted_indices]
    unique_features, unique_counts = np.unique(feature_vector, return_counts=True)
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2
    
    total_size = len(target_vector)
    targets_cumsum = np.cumsum(target_sorted)
    left_sizes = np.arange(1, total_size)
    left_cumsum = targets_cumsum[:-1]
    right_sizes = total_size - left_sizes
    right_cumsum = targets_cumsum[-1] - left_cumsum

    p0_left = left_cumsum / left_sizes
    p1_left = 1 - p0_left
    p0_right = (right_cumsum / right_sizes)
    p1_right = 1 - p0_right
    H_left = 1 - p1_left ** 2 - p0_left ** 2
    H_right = 1 - p1_right ** 2 - p0_right ** 2

    ginis = -(left_sizes / total_size) * H_left - (right_sizes / total_size) * H_right
    ginis = ginis[np.cumsum(unique_counts)[:-1] - 1]
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        # if np.all(sub_y != sub_y[0]):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        # for feature in range(1, sub_X.shape[1]):
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # ratio[key] = current_count / current_click
                    ratio[key] = current_click / current_count
                # sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                # feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            # if len(feature_vector) == 3:
            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                split = feature_vector < threshold
                if self._min_samples_leaf is None or (np.sum(split) >= self._min_samples_leaf
                                                      and np.sum(~split) >= self._min_samples_leaf):
                    feature_best = feature
                    gini_best = gini
    
                    if feature_type == "real":
                        threshold_best = threshold
                    # elif feature_type == "Categorical":
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0], 
                                                  filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError
        
        if feature_best is None or depth == self._max_depth or (self._min_samples_split is not None and self._min_samples_split > len(sub_y)):
            node["type"] = "terminal"
            # node["class"] = Counter(sub_y).most_common(1)
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        # self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]
        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_depth(self):
        return len(self._tree)


class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, base_model_type=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, n_quantiles=10):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)
        self.n_quantiles = n_quantiles

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_loss = None, None, float('inf')

        for feature in range(X.shape[1]):
            feature_vector = X[:, feature]
            thresholds = np.quantile(feature_vector, q=np.linspace(0, 1, self.n_quantiles + 1)[1:-1])
            for threshold in thresholds:
                left_mask = feature_vector <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
                    continue

                loss_left = self._compute_split_loss(X[left_mask], y[left_mask])
                loss_right = self._compute_split_loss(X[right_mask], y[right_mask])

                split_loss = (len(y[left_mask]) / len(y)) * loss_left + (len(y[right_mask]) / len(y)) * loss_right

                if split_loss < best_loss:
                    best_feature, best_threshold, best_loss = feature, threshold, split_loss

        return best_feature, best_threshold, best_loss

    def _compute_split_loss(self, X, y):
        if len(y) == 0:
            return 0
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        return mean_squared_error(y, predictions)

    def _fit_node(self, sub_X, sub_y, node, depth):
        if depth == self._max_depth or len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return

        feature, threshold, loss = self._find_best_split(sub_X, sub_y)

        if feature is None:
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature
        node["threshold"] = threshold

        left_mask = sub_X[:, feature] <= threshold
        right_mask = ~left_mask

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]

        feature_split = node["feature_split"]
        if x[feature_split] <= node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
