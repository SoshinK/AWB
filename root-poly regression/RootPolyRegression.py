import numpy as np

if __name__ == "__main__":
    pass

RP_REGRESSION_TERMS_P1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
RP_REGRESSION_TERMS_P2 = np.concatenate((RP_REGRESSION_TERMS_P1, np.array([[1 / 2, 1 / 2, 0], 
                                             [1 / 2, 0, 1 / 2], 
                                   [0, 1 / 2, 1 / 2]])))
RP_REGRESSION_TERMS_P3 = np.concatenate((RP_REGRESSION_TERMS_P2, np.array([ [1 / 3, 2 / 3, 0], 
                                             [0, 1 / 3, 2 / 3], 
                                             [1 / 3, 0, 2 / 3], 
                                             [2 / 3, 1 / 3, 0], 
                                             [0, 2 / 3, 1 / 3], 
                                             [2 / 3, 0, 1 / 3], 
                                             [1 / 3, 1 / 3, 1 / 3] ])))
RP_REGRESSION_TERMS_P4 = np.concatenate((RP_REGRESSION_TERMS_P3, np.array([ [3 / 4, 1 / 4, 0],
                                            [3 / 4, 0, 1 / 4],
                                            [1 / 4, 3 / 4, 0],
                                            [0, 3 / 4, 1 / 4],
                                            [1 / 4, 0, 3 / 4],
                                            [0, 1 / 4, 3 / 4],
                                            [2 / 4, 1 / 4, 1 / 4],
                                            [1 / 4, 2 / 4, 1 / 4],
                                            [1 / 4, 1 / 4, 2 / 4] ])))
RP_REGRESSION_POLYTERMS = np.array([RP_REGRESSION_TERMS_P1, RP_REGRESSION_TERMS_P2, RP_REGRESSION_TERMS_P3, RP_REGRESSION_TERMS_P4])

#! Input:
#!     X - array 3xN
#!     power - power of polynoms
#! Output:
#!     PxN array, where P - amount of root-polynomial terms
def get_rp_features(X, power):
    terms = RP_REGRESSION_POLYTERMS[0]
    if power == 2: terms = RP_REGRESSION_POLYTERMS[1]
    if power == 3: terms = RP_REGRESSION_POLYTERMS[2]
    if power == 4: terms = RP_REGRESSION_POLYTERMS[3]
    features = []
    for term in terms:
        features.append(np.prod(np.power(X.T, term), axis = 1))
    return np.array(features)

def q_rp_features(power):
    if power == 2: return 6
    if power == 3: return 13
    if power == 4: return 22
    else: return 3