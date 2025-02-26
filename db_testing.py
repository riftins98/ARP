import sqlite3
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
from arp_module import ARP
from algorithm1 import find_explanations
from mineArps import ARPMiner


class CrimeAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = ["ID", "DateAccour", "AreaNumber", "Type"]
        self.miner = ARPMiner(theta=0.01, delta=5, lambda_=0.01, Delta=5)

        # Define weights for attributes
        self.weights = {
            "ID": 0.1,
            "DateAccour": 0.3,  # Higher weight for temporal patterns
            "AreaNumber": 0.3,  # Important for spatial patterns
            "Type": 0.3
        }

        # Define distance functions
        self.distance_functions = {
            "ID": lambda x, y: 0.0 if x == y else 1.0,

            # Date distance (MM/YYYY format)
            "DateAccour": lambda x, y: self._date_distance(x, y),

            # Area number distance
            "AreaNumber": lambda x, y: abs(int(x) - int(y)) / 6.0,  # Normalized by max area number

            # Crime type distance
            "Type": lambda x, y: self._crime_type_distance(x, y)
        }

    def _date_distance(self, date1: str, date2: str) -> float:
        """Calculate normalized distance between two dates in MM/YYYY format"""
        try:
            m1, y1 = map(int, date1.split('/'))
            m2, y2 = map(int, date2.split('/'))
            months_diff = abs((y2 - y1) * 12 + (m2 - m1))
            return min(months_diff / 24.0, 1.0)  # Normalize to 2 years max
        except:
            return 1.0

    def _crime_type_distance(self, type1: str, type2: str) -> float:
        """Calculate distance between crime types"""
        if type1 == type2:
            return 0.0

        # Define crime categories
        theft_related = { "BATTERY - SIMPLE ASSAULT" } #'VEHICLE - STOLEN', 'BIKE - STOLEN', 'SHOPLIFTING-GRAND THEFT',

        # If crimes are in the same category
        if type1 in theft_related and type2 in theft_related:
            return 0.5

        return 1.0

    def fetch_data(self) -> List[Dict]:
        """Fetch data from SQLite database"""
        try:
            conn = sqlite3.connect('crimes.db')
            query = "SELECT ID, DateAccour, AreaNumber, Type FROM Crime"
            df = pd.read_sql_query(query, conn)

            # Convert to list of dictionaries
            records = df.to_dict('records')
            conn.close()
            return records

        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def analyze_patterns(self, question: Dict):
        """Analyze patterns and find explanations"""
        try:
            # Fetch data
            R = self.fetch_data()
            if not R:
                raise Exception("No data retrieved from database")

            # Mine patterns
            print("Mining patterns...")
            discovered_patterns = self.miner.mine_patterns(R, psi=4, schema=self.schema)
            seen = set()
            unique_patterns = []

            for arp in discovered_patterns:
                key = (frozenset(arp.F), tuple(arp.V), arp.agg, tuple(arp.A), arp.M)
                if key not in seen:
                    seen.add(key)
                    unique_patterns.append(arp)
            print(f"Discovered {len(unique_patterns)} patterns")

            # Find explanations
            print("\nFinding explanations...")
            explanations , total_relevant_patterns, total_tuples_searched = find_explanations(
                user_question=(question['Q'], R, question['t'], question['dir']),
                arps=unique_patterns,
                schema=self.schema,
                weights=self.weights,
                distance_functions=self.distance_functions,
                k=5,
                theta=0.01,
                delta=10,
                lambda_=0.01,
                Delta=10
            )

            return unique_patterns, explanations, total_relevant_patterns, total_tuples_searched

        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            return None, None

    def print_results(self, patterns, explanations, total_relevant_patterns, total_tuples_searched):
        """Print analysis results"""
        print("\nPattern Analysis Results:")

        if patterns:
            print("\nDiscovered Patterns:")
            for i, pattern in enumerate(patterns, 1):
                print(f"\nPattern {i}:")
                print(f"Grouping attributes: {pattern.F}")
                print(f"Predictive attributes: {pattern.V}")
                print(f"Pattern type: {pattern.M}")
                print(f'\nFount {total_relevant_patterns} relevant patterns')
                print(f'\nScored total of {total_tuples_searched} tuples')

        if explanations:
            print("\nTop Explanations:")
            for rank, (score, explanation) in enumerate(explanations, 1):
                print(f"\nRank {rank} (score: {score:.2f}):")
                for attr, value in explanation.items():
                    print(f"  {attr}: {value}")


def main():
    # Database path
    db_path = 'crimes.db'  # Update this path

    # Initialize analyzer
    analyzer = CrimeAnalyzer(db_path)

    # Example question: Why are there many vehicle thefts in area 07?
    question1 = {
        "Q": {"group_by": ["DateAccour", "AreaNumber"]},
        "t": ("02/2021", "07", 432),  # Example tuple
        "dir": "low",
        "description": "Why the crime in 02/2021 in area 07 is so low?"
    }

    # Analyze patterns and find explanations


    question2 = {
        "Q": {"group_by": ["DateAccour", "AreaNumber", "Type"]},
        "t": ("01/2024", "01", "BATTERY - SIMPLE ASSAULT", 276),  # Example tuple
        "dir": "high",
        "description": "Why the 'BATTERY - SIMPLE ASSAULT' crime in 01/2024 in area 01 is so high?"
    }

    while True:
        print("Welcome to the Crime Analyzer!")
        print("This is implemented by the CAPE and ARP algorithms.")
        print("you can analyze the following questions:")
        print(f'1) {question1["description"]}')
        print(f'2) {question2["description"]}')
        choice = input("Enter the number of the question you want to analyze: ")
        if choice == '1':
            patterns1, explanations1 , total_relevant_patterns, total_tuples_searched = analyzer.analyze_patterns(question1)
            if patterns1 and explanations1:
                analyzer.print_results(patterns1, explanations1, total_relevant_patterns, total_tuples_searched)
            else:
                print("Analysis failed or no patterns found.")
        elif choice == '2':
            patterns2, explanations2 , total_relevant_patterns, total_tuples_searched = analyzer.analyze_patterns(question2)
            if patterns2 and explanations2:
                analyzer.print_results(patterns2, explanations2, total_relevant_patterns, total_tuples_searched)
            else:
                print("Analysis failed or no patterns found.")
        elif choice.lower() == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")




    # Additional analysis examples:
    # print("\nBasic Statistics:")
    # data = pd.DataFrame(analyzer.fetch_data())
    # if not data.empty:
    #     print(f"Total records: {len(data)}")
    #     print("\nCrime distribution by area:")
    #     print(data['AreaNumber'].value_counts().head())
    #     print("\nCrime type distribution:")
    #     print(data['Type'].value_counts().head())
    #     print("\nTemporal distribution:")
    #     print(data['DateAccour'].value_counts().head())


if __name__ == "__main__":
    main()