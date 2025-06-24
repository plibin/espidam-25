import numpy as np
import itertools

def get_strategy_combinations() -> np.ndarray:
    """
    Returns a list of all strategy combinations.

    """
    general_strategy = ['SI',    # symptomatic isolation
                        'RS',    # reactive screening
                        'RS_A']  # repetitive screening
    n_test_week = [1, 2]
    threshold_class = [2, 4, 8, 1000]
    threshold_school = [10, 100, 1000]
    combinations = list(itertools.product(general_strategy, 
                                          n_test_week, 
                                          threshold_class, 
                                          threshold_school))
    strategy_combinations = [{'general_strategy': str(gs), 
                              'n_test_week': str(ntw), 
                              'threshold_class': str(tc), 
                              'threshold_school': str(ts)} \
                                for gs, ntw, tc, ts in combinations]
    
    return strategy_combinations

print(get_strategy_combinations())
