#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:00:33 2018

@author: jeremiasknoblauch

Description: Implements an AR-approach where each location gets only its own
past as regressors, and where we do NOT allow for spatial relations. Basically,
it amounts to the 'independent' model in Murphy.
"""

import numpy as np
from scipy import linalg
from scipy import misc

from probability_model import ProbabilityModel
from BVAR_NIG import BVARNIG

class BARNIG(ProbabilityModel):
    """This creates an object based on and similar to the BVARNIG object.
    In particular, each location is treated as completely independent from
    all other locations. No endogeneous regressors from other locations are
    allowed for [though we might want to add that feature in time], and the
    variance is modelled as independent
    
    Attributes:
        S1, S2                      spatial dimensions
        joint_log_probabilities     P(y_1:t, r_t, m_t)
        location_models             S1*S2 BVAR_NIG objects, one for each 
                                        location
    
    """
    
    def __init__(self, prior_a_list, prior_b_list, 
                 lags_list,
                 S1, S2, 
                 prior_mean_beta_list=None, prior_var_beta_list=None,
                 prior_mean_scale_list=None, prior_var_scale_list=None,
                 non_spd_alerts_list = None):
        """Creates the object by creating a BVAR object per location. For 
        arguments that were not supplied, it creates default-options before
        doing so."""
        
        """STEP 1: Sanity check: Is S1*S2 = len(lag_list)?"""
        if not S1*S2 == len(lags_list):
            print("ERROR! S1, S2 do not match dimension of lags_list")
            return
        else:
            """If it is not, save everything"""
            self.S1, self.S2 = S1, S2
            self.lags_list = lags_list
        
        """STEP 2: Construct priors that were not supplied (convenience 
        feature)"""
        
        """STEP 2.1: do it for the coefficient mean"""
        if prior_mean_beta_list is None and prior_mean_scale_list is None:
            self.prior_mean_beta_list = []
            for lag in lags_list:
                self.prior_mean_beta_list.append(np.zeros(lag+1))
        elif not prior_mean_beta_list is None:
            self.prior_mean_beta_list = prior_mean_beta_list
        elif not prior_mean_scale_list is None:
            self.prior_mean_beta_list = []
            for (lag, location)  in zip(lags_list, range(0, self.S1*self.S2)):
                self.prior_mean_beta_list.append(
                        np.ones(lag+1)*prior_mean_scale_list[location])
            
        """STEP 2.2: Do it for the coef var"""
        if prior_var_beta_list is None and prior_var_scale_list is None:
            self.prior_var_beta_list = []
            for lag in lags_list:
                self.prior_var_beta_list.append(np.identity(lag+1)*100)
        elif not prior_var_beta_list is None:
            #print("reached")
            self.prior_var_beta_list = prior_var_beta_list
            #print(self.prior_var_beta_list)
        elif not prior_var_scale_list is None:
            self.prior_var_beta_list = []
            for (lag, location)  in zip(lags_list, range(0, self.S1*self.S2)):
                self.prior_var_beta_list.append(
                        np.identity(lag+1)*prior_var_scale_list[location])
            
        """STEP 2.3: Do it for the spd alerts"""
        self.non_spd_alerts_list = [False]*self.S1*self.S2
        
        """STEP 2.4: Do it for the nbh sequences"""
        self.nbh_sequence_list = []
        self.res_sequence_list = []
        for lag in lags_list:
            self.nbh_sequence_list.append([0]*lag)
            self.res_sequence_list.append([0]*lag)
            
        """STEP 3: Create the list of BVAR objects, one for each location"""
        self.location_models = []
        for location in range(0, self.S1*self.S2):
            #print(self.prior_var_beta_list[location])
            self.location_models.append(
                BVARNIG(prior_a_list[location], prior_b_list[location],
                        S1=1, S2=1, 
                        prior_mean_beta = self.prior_mean_beta_list[location],
                        prior_var_beta = self.prior_var_beta_list[location],
                        nbh_sequence=self.nbh_sequence_list[location],
                        restriction_sequence=self.res_sequence_list[location]
                        )
                )
        
        """STEP 4: Create additional things that are not input-specific"""
        self.retained_run_lengths = np.array([0,0])
        self.joint_log_probabilities = 1
        self.has_lags = True
        self.lag_length = max(self.lags_list) #choose the max lag length from the list
        self.auto_prior_update = False
        self.exo_bool = False
        self.model_log_evidence = -np.inf
        self.nbh_sequence = -1
        
        
        
        
    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior,
            padding_columns_computeXX = None, padding_column_get_x_new = None):
        """Initialize the model (i.e. t=1) with some inputs from the 
        containing Detector object. The padding_column arguments are only 
        needed for the demo Csurf object.
        
        NOTE:    The X_exo-arguments are not relevant for BAR_NIG (at this stage)
        """
        
        print("Initializing BAR object")
        
        """STEP 1: Get relevant variables and reshape them + pass them to 
        individual BVARs for initialization"""
        Y2 = Y_2.flatten()
        
        """NOTE: Detector passes a chunck with length lag_length + 1, which 
        corresponds to the maximum of all lag lengths + 1. We need to cut off 
        from X_endo's left hand side the difference (max-lag - this-model-lag)
        before initializing that model"""
        
        
        for (lag, location) in zip(self.lags_list, range(0, self.S1*self.S2)):
                      
            """NOTE: since we will just sum up the outputs of all individual 
            location models' log_densities, we only want to pass the 'true'
            prior into one of the models. Otherwise we would multiply our
            joint log probability S1*S2 times with the prior at t=0"""

            X_endo_loc = X_endo[(self.lag_length - lag):,location]
            n = np.size(X_endo_loc)
            X_endo_loc = X_endo_loc.reshape(n,1)
            self.location_models[location].initialization(X_endo = X_endo_loc, 
                X_exo = None, Y_2 = Y2[location], X_exo_2 = None,
                cp_model = cp_model, model_prior=1,
                padding_columns_computeXX=None, 
                padding_column_get_x_new=None)
        
        
        """STEP 2: Get the model log_evidence, using the location models"""
        self.model_log_evidence = model_prior + np.sum(
            [loc_mod.model_log_evidence for loc_mod in self.location_models])
        
        """STEP 3: Get the joint log probabilities, using location models"""
        
        """Ensure that we do not get np.log(0)=np.inf by perturbation"""
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.000000000001
        else:   
            epsilon = 0
        
        """Compute joint log probs"""
        r_equal_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(0) + epsilon)) 
        r_larger_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(1)+ epsilon))   
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0]) 

        
    def evaluate_predictive_log_distribution(self, y, t):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by 
        virtue of the sufficient statistics.             
        The corresponding density is computed for all run-lengths and
        returned in a np array. Here, this means we compute the log densities
        for each individual location model, and then sum them up (due to 
        independence assumption across locations)"""
        
        """STEP 1: Bring everything we need into the right form"""                                                    
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = np.zeros(shape=run_length_num)
        
        """STEP 2: Extract log densities of each model and add them together"""
        for location in range(0, self.S1*self.S2):
            log_densities += (self.location_models[location].
                evaluate_predictive_log_distribution(y[location],t))
        
        """STEP 3: Return the result"""
        return log_densities
    
    
    def evaluate_log_prior_predictive(self, y, t):
        """use only the prior specs of BVARNIG object to get predictive prob"""
        prior_prob = 0
        for location in range(0, self.S1*self.S2):
            prior_prob += (self.location_models[location].
                evaluate_log_prior_predictive(y[location],t))
        return prior_prob
    
    
    def save_NLL_fixed_pars(self, y,t):
        """get the NLL for each BVAR object and simply add it together"""
        y = y.flatten()
        helper = np.zeros(self.retained_run_lengths.shape[0])
        for location in range(0,self.S1*self.S2):
            #NOTE: They should all have the same length since lag length is
            #      the same across al location models
            self.location_models[location].save_NLL_fixed_pars(y[location],t)
            helper+= (self.location_models[location].
                      one_step_ahead_predictive_log_probs_fixed_pars)
        self.one_step_ahead_predictive_log_probs_fixed_pars = helper
    
    
    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1, t, 
                                        padding_column_tm1 = None,
                                        padding_column_t = None, 
                                        r_evaluations = None):
        """Takes the next observation, *y*, at time *t* and updates the
        sufficient statistics, means & vars corresponding to all potential 
        run-lengths r=0,1,...,t-1,>t-1. This is simply done inside the model.
        """
        
        """STEP 1: Update all the quantities inside the location models"""
        y_t = y_t.flatten()
        y_tm1 = y_tm1.flatten()
        for location in range(0, self.S1*self.S2):
            self.location_models[location].update_predictive_distributions(
                y_t[location], y_tm1[location], x_exo_t, x_exo_tp1, t, 
                padding_column_tm1 = None, padding_column_t = None, 
                r_evaluations = None)
            
        """STEP 2: change what we need to adapt inside BAR object"""
        self.retained_run_lengths =  self.retained_run_lengths + 1 
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)
        
    
    
    def trimmer(self, kept_run_lengths):
        """Trim the relevant quantities for the BAR NIG model"""
        self.joint_log_probabilities = (
                    self.joint_log_probabilities[kept_run_lengths])
        self.retained_run_lengths = (
                    self.retained_run_lengths[kept_run_lengths])
        self.model_log_evidence = misc.logsumexp(
                        self.joint_log_probabilities )
        
        """Trim all quantities for the location-specific models"""
        for location in range(0, self.S1*self.S2):
            self.location_models[location].trimmer(kept_run_lengths, 
                                BAR_submodel = True)
            
            
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors 
        at time point t, for all possible run-lengths. Get a prediction
        for each location, and then tie them all together in one array of the
        correct dimensions"""
        num_retained_run_lengths = np.size(self.retained_run_lengths)
        post_mean = np.zeros((num_retained_run_lengths, 
                              self.S1*self.S2))
        for location in range(0, self.S1*self.S2):
            #print(self.location_models[location].
            #         get_posterior_expectation(t))
            post_mean[:,location] = (self.location_models[location].
                get_posterior_expectation(t).reshape(num_retained_run_lengths))
        
        return post_mean


    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at 
        time point t, for all possible run-lengths. Get a prediction
        for each location, and then tie them all together in one array of the
        correct dimensions"""
        num_retained_run_lengths = np.size(self.retained_run_lengths)
        post_var = np.zeros((num_retained_run_lengths, 
                                     self.S1*self.S2, self.S1*self.S2))
        
        """Note: This makes the diag non-zero and the rest 0"""
        for location in range(0, self.S1*self.S2):
            post_var[:,location, location] = (self.location_models[location].
                get_posterior_variance(t).reshape(num_retained_run_lengths))
        
        return post_var

