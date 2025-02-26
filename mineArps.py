import itertools
from collections import defaultdict
import numpy as np
from scipy import stats
from regression_models import RegressionModel
from arp_module import ARP


class ARPMiner:
    def __init__(self, theta=0.8, delta=2, lambda_=0.5, Delta=2):
        """
        Initialize ARPMiner with thresholds.

        Args:
            theta (float): Goodness-of-fit threshold
            delta (int): Minimum support threshold
            lambda_ (float): Global confidence threshold
            Delta (int): Global support threshold
        """
        self.theta = theta
        self.delta = delta
        self.lambda_ = lambda_
        self.Delta = Delta
        self.supported_aggs = ["count", "sum", "avg", "min", "max"]
        self.supported_models = ["constant", "linear"]

    def mine_patterns(self, R, psi, schema):
        """
        Implementation of Algorithm 2: ARP-mine pattern discovery

        Args:
            R: Input relation
            psi: Maximum size of group-by attributes
            schema: Schema of R

        Returns:
            set: Discovered patterns that hold globally
        """
        P = set()  # Patterns that hold globally
        C = set()  # Candidate (F, V) we have considered
        groupSizes = {}  # Map of G to |π_G(R)|

        # Step 4: for i ∈ {2,...,ψ}
        for i in range(2, psi + 1):
            # Step 5: for each G ⊂ R and |G| = i
            for G in itertools.combinations(schema, i):
                # Step 6: A_agg = R - G
                A_agg = [attr for attr in schema if attr not in G]

                # Step 7-8: Execute aggregation query
                D = self._execute_query(R, G, A_agg)

                # Step 9: Insert into groupSizes
                groupSizes[G] = len(D)

                # Step 10: Detect functional dependencies
                FDs = self._detect_FDs(groupSizes, G)

                # Step 11: Explore sort orders
                P, C = self._explore_sort_orders(G, D, C, P, R)

        return P

    def _explore_sort_orders(self, G, D, C, P, R):
        """
        Implementation of Algorithm 5: Explore sort orders
        """
        # Convert G to tuple for consistent handling
        G = tuple(sorted(G))  # Sort to ensure consistent ordering

        # Step 1: for each permutation S of G
        for S in itertools.permutations(G):
            # Step 2-4: Sort D by S
            D_sort = sorted(D, key=lambda x: tuple(x[attr] for attr in S))

            # Step 5: Check each F,V pair
            for i in range(1, len(S) + 1):
                F = S[:i]
                V = tuple(attr for attr in G if attr not in F)

                if (F, V) not in C and self._is_prefix(F, S):
                    C.add((F, V))

                    # Create and test patterns
                    for M in self.supported_models:
                        pattern = ARP(F=list(F), V=list(V), agg="count", A=["*"], M=M)
                        if self._validate_pattern(pattern, R) and self._fit_pattern(pattern, D_sort):
                            P.add(pattern)

        return P, C

    def _fit_pattern(self, P, D_sort):
        """
        Implementation of Algorithm 6: Checking whether pattern holds
        """
        frag_good = set()
        frag_supp = set()
        f = None
        D_f = []
        h_f = {}
        regression_models = {}

        for t in D_sort:
            f_cur = tuple(self._convert_value(t[attr], attr) for attr in P.F)
            if f_cur == f:
                # Collect data for current fragment
                v_key = tuple(self._convert_value(t[attr], attr) for attr in P.V)
                if P.A[0] == "*":
                    h_f[v_key] = h_f.get(v_key, 0) + 1
                else:
                    try:
                        h_f[v_key] = float(t[P.agg])
                    except (ValueError, TypeError):
                        h_f[v_key] = 0
                D_f.append(t)
            else:
                # Process previous fragment if it exists
                if f is not None and len(D_f) >= self.delta:
                    frag_supp.add(f)
                    # Apply regression and check GoF
                    model = RegressionModel(P.M)
                    X = list(h_f.keys())
                    y = list(h_f.values())

                    try:
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        gof = model.goodness_of_fit(y, y_pred)
                        if gof > self.theta:
                            frag_good.add(f)
                        regression_models[f] = model
                    except Exception as e:
                        print(f"Error in regression: {str(e)}")

                # Start new fragment
                f = f_cur
                D_f = [t]
                h_f = {}
                v_key = tuple(self._convert_value(t[attr], attr) for attr in P.V)
                if P.A[0] == "*":
                    h_f[v_key] = 1
                else:
                    try:
                        h_f[v_key] = float(t[P.agg])
                    except (ValueError, TypeError):
                        h_f[v_key] = 0

        # Process last fragment
        if f is not None and len(D_f) >= self.delta:
            frag_supp.add(f)
            model = RegressionModel(P.M)
            X = list(h_f.keys())
            y = list(h_f.values())

            try:
                model.fit(X, y)
                y_pred = model.predict(X)
                gof = model.goodness_of_fit(y, y_pred)
                if gof > self.theta:
                    frag_good.add(f)
                regression_models[f] = model
            except Exception as e:
                print(f"Error in regression: {str(e)}")

        # Check global conditions
        P.model=regression_models
        # P.localFrag=frag_good  # TODO was added
        if len(frag_good) >= self.Delta:
            confidence = len(frag_good) / len(frag_supp) if frag_supp else 0
            return confidence >= self.lambda_
        return False

    def _validate_pattern(self, pattern, R):
        """
        Validate a pattern before mining
        """
        # Check if pattern attributes are valid
        if not pattern.F or not all(attr in R[0].keys() for attr in pattern.F):
            return False

        if pattern.V and not all(attr in R[0].keys() for attr in pattern.V):
            return False

        # Check for overlapping attributes
        if set(pattern.F) & set(pattern.V):
            return False

        # Check aggregation
        if pattern.agg not in self.supported_aggs:
            return False

        # Check model type
        if pattern.M not in self.supported_models:
            return False

        return True

    def _convert_value(self, value, attr):
        """Convert value based on attribute type"""
        if attr == 'year':
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        return value

    def _get_fragment_data(self, R, F_values, F):
        """Get data for a specific fragment"""
        return [
            row for row in R
            if all(self._convert_value(row[attr], attr) == val
                   for attr, val in zip(F, F_values))
        ]

    def _is_prefix(self, F, S):
        """Check if F is a prefix of sort order S"""
        return all(F[i] == S[i] for i in range(len(F)))

    def _detect_FDs(self, groupSizes, G):
        """Detect functional dependencies using group sizes"""
        FDs = set()
        G_size = groupSizes[G]

        for i in range(1, len(G)):
            for subset in itertools.combinations(G, i):
                if subset in groupSizes and groupSizes[subset] == G_size:
                    FDs.add((G, subset))

        return FDs

    def _execute_query(self, R, G, A_agg):
        """
        Execute aggregation query Q = SELECT G, agg(A1),... FROM R GROUP BY G

        Args:
            R: Input relation
            G: Group-by attributes
            A_agg: Attributes to aggregate

        Returns:
            list: Results of aggregation query
        """
        grouped_data = defaultdict(list)

        # Group the data
        for row in R:
            key = tuple(self._convert_value(row[attr], attr) for attr in G)
            grouped_data[key].append(row)

        results = []
        for key, group in grouped_data.items():
            result = dict(zip(G, key))

            # Apply each aggregation function to each attribute
            for attr in A_agg:
                for agg in self.supported_aggs:
                    if agg == "count":
                        if attr == "*":
                            # COUNT(*) counts all rows
                            result[f"{agg}_{attr}"] = len(group)
                        else:
                            # COUNT(attr) counts non-null values
                            result[f"{agg}_{attr}"] = sum(1 for row in group if row.get(attr) is not None)

                    elif agg == "sum":
                        try:
                            # Sum numeric values
                            values = [float(row[attr]) for row in group if attr in row]
                            result[f"{agg}_{attr}"] = sum(values)
                        except (ValueError, TypeError):
                            continue

                    elif agg == "avg":
                        try:
                            # Average numeric values
                            values = [float(row[attr]) for row in group if attr in row]
                            if values:
                                result[f"{agg}_{attr}"] = sum(values) / len(values)
                        except (ValueError, TypeError):
                            continue

                    elif agg == "min":
                        try:
                            # Minimum value
                            values = [row[attr] for row in group if attr in row]
                            if values:
                                result[f"{agg}_{attr}"] = min(values)
                        except (ValueError, TypeError):
                            continue

                    elif agg == "max":
                        try:
                            # Maximum value
                            values = [row[attr] for row in group if attr in row]
                            if values:
                                result[f"{agg}_{attr}"] = max(values)
                        except (ValueError, TypeError):
                            continue

            results.append(result)

        return results