#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:16:38 2019

@author: tawehbeysolow
"""

from collections import defaultdict
import numpy as np

class EligibilityTrace(object):
    """class containing logic for SARSA-lambda eligibility traces
        this is basically a wrapper for a dict that 
            1) clips its values to lie in the interval [0, 1]
            2) updates all values by a decay constant and throws out those
                that fall below some threshold
    """
    def __init__(self, decay, threshold):
        self.decay = decay
        self.threshold = threshold
        self.data = defaultdict(float)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = np.clip(val, 0, 1)

    def iteritems(self):
        return self.data.iteritems()

    def update(self):
        for key in self.data.keys():
            if self.data[key] < self.threshold:
                del self.data[key]
            else:
                self.data[key] = self.data[key] * self.decay


class SARSA(Agent):
    """impementation of SARSA lambda algorithm.
        class SARSA is equivilant to this with lambda = 0, but 
        we seperate the two out because
            1) it's nice to juxtapose the two algorithms side-by-side
            2) SARSA lambda incurrs the overhead of maintaining
                eligibility traces
        note that the algorithm isn't explicitly parameterized with lambda.
            instead, we provide a decay rate and threshold. On each iteration,
            the decay is applied all rewards in the eligibility trace. Those 
            past rewards who have decayed below the threshold are dropped
    """
    def __init__(self, featureExtractor, max_gradient, epsilon=0.5, gamma=0.993, stepSize=None, threshold=0.1, decay=0.98):
        super(SARSA, self).__init__(featureExtractor, epsilon, gamma, stepSize, max_gradient)
        self.eligibility_trace = EligibilityTrace(decay, threshold)

    def update_q_matrix(self, state, action, reward, newState):
        """performs a SARSA update. Leverages the eligibility trace to update 
            parameters towards sum of discounted rewards
        """
        self.eligibility_trace.update()
        prediction = self.getQ(state, action)
        newAction = None
        target = reward
        for f, v in self.featureExtractor.get_features(state, action).iteritems():
            self.eligibility_trace[f] += v

        if newState != None:
            newAction = self.takeAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = self.getStepSize(self.numIters) * (prediction - target)
        # clip gradient - TODO EXPORT TO UTILS?
        update = max(-self.max_gradient, update) if update < 0 else min(self.max_gradient, update)

        for key, eligibility in self.eligibility_trace.iteritems():
            self.weights[key] -= update * eligibility
        return newAction


