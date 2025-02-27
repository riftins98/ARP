import math
import scipy.stats as stats


class ARP:
    """
    Aggregate Regression Pattern class as defined in the paper.
    """

    def __init__(self, F, V, agg, A, M):
        """
        Initialize an ARP.

        Args:
            F (list): Partition attributes
            V (list): Predictor attributes
            agg (str): Aggregation function
            A (list): Attributes to aggregate
            M (str): Regression model type
        """
        self.F = F
        self.V = V
        self.agg = agg
        self.A = A
        self.M = M
        self.retrival=None
        self.frags=None
        self.localFrag=None
        self.norm=None

    def __repr__(self):
        return f"ARP(F={self.F}, V={self.V}, agg={self.agg}, A={self.A}, M={self.M})"


    def frag(self, R):
        """
        Get distinct values for attributes F in relation R.

        Args:
            R (list): List of dictionaries representing the relation
            F (list): List of attribute names

        Returns:
            list: List of distinct F-value combinations
        """
        if self.frags is not None:
            return self.frags
        fragments = set()
        for row in R:
            key = tuple(row[attr] for attr in self.F)
            fragments.add(key)
        self.frags=list(fragments)
        return list(fragments)

    def retrieval_query(self, R):
        """
        Execute retrieval query for each fragment, maintaining original logic.

        Args:
            R (list): Input relation

        Returns:
            dict: Mapping from fragment values to query results
        """
        if self.retrival is not None:
            return self.retrival
        fragments = self.frag(R)
        results = {}

        for f in fragments:
            # Direct filtering with minimal function calls
            matching_rows = [
                row for row in R
                if all(row[attr] == f_val for attr, f_val in zip(self.F, f))
            ]

            # Fast grouping
            grouped = {}
            for row in matching_rows:
                v_key = tuple(row[attr] for attr in self.V)

                # Efficient aggregation value collection
                value = 1 if self.A[0] == "*" else row[self.A[0]]

                if v_key not in grouped:
                    grouped[v_key] = [value]
                else:
                    grouped[v_key].append(value)

            # Optimized aggregation
            agg_results = []
            for v_key, values in grouped.items():
                result = {
                    **dict(zip(self.F, f)),
                    **dict(zip(self.V, v_key))
                }

                # Inline aggregation
                if self.agg == "count":
                    result[self.agg] = len(values)
                elif self.agg == "sum":
                    result[self.agg] = sum(values)
                elif self.agg == "avg":
                    result[self.agg] = sum(values) / len(values)

                agg_results.append(result)
            results[f] = agg_results
        self.retrival=results

        return results



    def holds_locally(self, fragment_data, theta, delta):
        """
        Check if pattern holds locally for fragments.

        Args:
            P (ARP): The pattern
            fragment_data (dict): Fragment data from retrieval query
            theta (float): Goodness-of-fit threshold
            delta (int): Support threshold

        Returns:
            set: Fragments where pattern holds locally
        """
        if self.localFrag is not None:
            return self.localFrag
        local_fragments = set()


        for f, results in fragment_data.items():
            if len(results) < delta:
                continue

            # Extract values for regression
            values = [r[self.agg] for r in results]

            # Calculate goodness of fit
            if self.M == "constant":
                mean = sum(values) / len(values)
                predicted = [mean] * len(values)
                gof = goodness_of_fit(values, predicted)

                if gof >= theta:
                    local_fragments.add(f)
        self.localFrag=local_fragments
        return local_fragments

    def holds_globally(self,fragment_data, fragments ,local_fragments, Delta, lambda_):
        # Check global conditions
        frag_good = len(local_fragments)
        frag_supp = len([f for f in fragments if len(fragment_data[f]) >= Delta])

        return frag_good >= Delta and (frag_good / frag_supp if frag_supp else 0) >= lambda_

    def is_relevant_pattern(self, user_question, theta, delta):
        """
        Check if pattern is relevant for user question.

        Args:
            P (ARP): The pattern
            user_question (tuple): (Q, R, t, dir)
            theta (float): Goodness-of-fit threshold
            delta (int): Support threshold

        Returns:
            bool: True if pattern is relevant
        """
        Q, R, t, dir = user_question
        G = Q["group_by"]

        # Check if F ∪ V ⊆ G
        if not set(self.F + self.V).issubset(set(G)):
            return False

        # Check local hold for fragment containing t
        # fragment_data = self.retrieval_query(R)
        # local_fragments = self.holds_locally(fragment_data, theta, delta)

        # return len(local_fragments) > 0
        return True

    def is_refinement(self, P):
        """
        Check if P_prime is a refinement of P.

        Args:
            P_prime (ARP): Potential refinement pattern
            P (ARP): Original pattern
        Returns:
            bool: True if P_prime is a refinement of P
        """
        # Check if F' ⊇ F
        return set(self.F).issuperset(set(P.F)) and set(self.V)==set(P.V)


    def is_relevant_tuple(self, Q, t, t_prime):
        for attr in self.F:
            attr_to_index = {attr: idx for idx, attr in enumerate(Q["group_by"])}
            if t[attr_to_index[attr]]!= t_prime[attr]:
                return False
        return True


def goodness_of_fit(actual, predicted):
    """
    Calculate goodness of fit using chi-square test.

    Args:
        actual (list): Actual values
        predicted (list): Predicted values

    Returns:
        float: Goodness of fit value
    """
    if not actual or len(actual) != len(predicted):
        return 0.0

    _, p_value = stats.chisquare(actual, predicted)
    return p_value




def score_explanation(E, user_question, schema, weights=None, distance_functions=None):
    """
    Calculate explanation score according to Definition 10:
    score(E) = (dev_P'(t') * isLow) / (d(t[G], t'[F', V]) * NORM)
    where NORM = π_agg(A)(σ_F=t[F]∧V=t[V](γF∪V,agg(A)(R)))
    """
    try:
        P, P_prime, t_prime = E["P"], E["P_prime"], E["t_prime"]
        Q, R, t, dir = user_question

        # Convert question tuple to dictionary
        t_dict = dict(zip(Q["group_by"], t[:-1]))  # Exclude count
        t_dict['agg'] = t[-1]  # Add count separately

        # Get regression prediction
        F_values = tuple(t_prime[attr] for attr in P_prime.F)
        if F_values not in P_prime.model:
            return 0.0


        # Calculate expected value
        try:
            V_values = tuple(t_prime[attr] for attr in P_prime.V)
            expected_value = P_prime.model[F_values].predict_func(V_values)
            # retrieval_results = P_prime.retrieval_query(R)
            # matching_results = [res for res_list in retrieval_results.values() for res in res_list if
            #                     all(res[attr] == t_prime[attr] for attr in P_prime.V)]
            # if matching_results:
            #     expected_value = sum(res[P_prime.agg] for res in matching_results) / len(matching_results)
            # else:
            #     expected_value = 0.0

        except (KeyError, ValueError):
            return 0.0

        # Calculate deviation
        actual_value = t_prime.get(P_prime.agg, 0)
        deviation = actual_value - expected_value

        # Calculate distance
        distance = calculate_distance(t_dict, t_prime, schema, weights, distance_functions)
        if distance == 0.0:
            distance = 1.0

        if P.norm is not None:
            norm = P.norm
        else:
            # Get matching tuples
            matching_tuples = [row for row in R
                               if all(row.get(attr) == t_dict.get(attr) for attr in P.F + P.V)]

            # Calculate aggregation value for matching tuples
            if matching_tuples:
                if P.agg == "count":
                    norm = len(matching_tuples)
                else:
                    norm = sum(float(row.get(P.A[0], 0)) for row in matching_tuples)
            else:
                norm = 1.0
            P.norm=norm

        # Calculate final score with NORM
        score = deviation / (distance * norm)

        if dir == "high":
            score1 = -score
        return score

    except Exception as e:
        print(f"Detailed error in score calculation: {str(e)}")
        return 0.0

def calculate_distance(t1, t2, schema, weights=None, distance_functions=None):
    """
    Calculate distance between tuples t1 and t2 according to the paper's formula,
    where T1 and T2 are the subschemas of attributes that exist in each tuple.

    Args:
        t1, t2: Tuples to compare
        schema: Complete schema of the relation
        weights: Dictionary of weights that sum to 1 across all attributes
        distance_functions: Dictionary of distance functions for each attribute
    """
    if weights is None:
        # Initialize weights that sum to 1
        weights = {attr: 1.0 / len(schema) for attr in schema}

    if distance_functions is None:
        distance_functions = {attr: lambda x, y: 0.0 if x == y else 1.0 for attr in schema}

    # Get the subschemas (attributes that exist in each tuple)
    T1 = set(t1.keys())
    T2 = set(t2.keys())

    # Get common attributes
    common_attrs = T1 & T2
    W=0.0
    set_attr = T1 | T2
    for attr in set_attr:
        if attr not in weights:
            continue
        W += weights[attr]



    if W == 0:
        return 1.0

    # Calculate sum of weighted squared distances
    sum_weighted_distances = 0.0

    # For each attribute in the union of T1 and T2
    for attr in (T1 | T2):
        if attr not in weights:
            continue
        w_A = weights[attr]

        # Calculate d_A^exists
        if attr in common_attrs:
            # Attribute exists in both tuples - use distance function
            d_A = distance_functions[attr](t1[attr], t2[attr])
        else:
            # Attribute exists in only one tuple
            d_A = 1.0

        sum_weighted_distances += w_A * (d_A ** 2)

    # Calculate final distance
    if W == 0:
        return sum_weighted_distances
    else:
        if (1.0 / W) * sum_weighted_distances < 0:
            return 1.0
    return math.sqrt((1.0 / W) * sum_weighted_distances)