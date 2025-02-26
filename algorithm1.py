from datetime import datetime
import heapq
from heapq import heappush, heappop
from math import trunc

from arp_module import score_explanation



def find_explanations(user_question, arps, schema, weights, distance_functions, k,
                      theta, delta, lambda_, Delta):
    Q, R, t, dir = user_question
    explanations = []
    seen_explanations = set()
    count = 0
    total_relevant_patterns = 0
    total_tuples_searched = 0


    non_relevant_patterns = []
    for P in arps:
        if not P.is_relevant_pattern(user_question, theta, delta):
            non_relevant_patterns.append(P)
            continue
        print(f"ARP is relevant: F = {P.F}, V = {P.V}, agg = {P.agg}, A = {P.A}, M = {P.M}" )
        tuple_per_ARP = total_tuples_searched
        total_relevant_patterns += 1
        refinements = {P_prime for P_prime in arps if P_prime.is_refinement(P) and P_prime not in non_relevant_patterns}
        seen = set()
        unique_refinements = []

        for arp in refinements:
            key = (frozenset(arp.F), tuple(arp.V), arp.agg, tuple(arp.A), arp.M)
            if key not in seen:
                seen.add(key)
                unique_refinements.append(arp)

        for P_prime in unique_refinements:
            retrieval_results = P_prime.retrieval_query(R)
            for fragment_key in retrieval_results:
                r={}
                fragment_data = retrieval_results[fragment_key]
                r[fragment_key]=fragment_data
                for t_prime in fragment_data:
                    total_tuples_searched += 1
                    # Check if explanation has any matching attributes
                    has_match = P.is_relevant_tuple(Q,t,t_prime)


                    if not P_prime.holds_locally(r,theta, delta):
                        has_match = True
                    if not has_match:
                        continue

                    expl_key = tuple(sorted(t_prime.items()))
                    if expl_key in seen_explanations:
                        continue

                    seen_explanations.add(expl_key)
                    explanation = {
                        "P": P,
                        "P_prime": P_prime,
                        "t_prime": t_prime
                    }

                    score = score_explanation(explanation, user_question, schema,
                                              weights, distance_functions)
                    updateExpl(score, explanation, explanations, k, count)
                    count += 1
        if tuple_per_ARP != total_tuples_searched:
            print(f"Found {total_tuples_searched - tuple_per_ARP} relevant patterns for this ARP")
        else:
            print("No relevant patterns found for this ARP")


    # Convert back to positive scores and sort
    result = []
    while explanations:
        score, _, expl = heappop(explanations)
        result.append((score, expl["t_prime"]))

    return list(reversed(result)), total_relevant_patterns, total_tuples_searched  # Reverse to get highest scores first

def updateExpl(score, explanation, explanations, k, count):
    if len(explanations) < k:
        # Use negative score to create a min-heap of highest scores
        heapq.heappush(explanations, (score, count, explanation))
    elif score > explanations[0][0]:  # Compare with the smallest score in heap
        heapq.heappop(explanations)

        heapq.heappush(explanations, (score, count, explanation))
    else:
        return