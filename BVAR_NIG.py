# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:30:37 2017
@author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)

Description: Implements Bayesian Linear Autoregression with the NIG model 
(i.e., spatial locations have iid errors with common variance)
"""

import numpy as np
#from scipy import stats 
from scipy import special
from scipy import linalg
from scipy import stats
import scipy
from probability_model import ProbabilityModel
from nearestPD import NPD
#from cp_probability_model import CpModel


class BVARNIG(ProbabilityModel):
    """The Bayesian Vector Autoregression model using past observations as 
    regressors in a specified neighbourhood. E.g., if the 4-neighbourhood is
    selected with lag length 1, then the mean of y_{t,i} is modelled as linear
    combination of observations y_{t-1, j} \in nb(i). Around the boundary,
    the neighbourhoods are 0-padded. 
    
    
    """
    
    def __init__(self, 
                 prior_a, 
                 prior_b, 
                 S1, 
                 S2, 
                 prior_mean_beta=None, 
                 prior_var_beta=None,
                 prior_mean_scale=0, 
                 prior_var_scale=100,
                 nbh_sequence=None, 
                 restriction_sequence = None,
                 intercept_grouping = None,
                 general_nbh_sequence = None,
                 general_nbh_restriction_sequence = None,
                 nbh_sequence_exo=None, 
                 exo_selection = None,
                 padding = 'overall_mean', 
                 auto_prior_update=False,
                 hyperparameter_optimization = "online",
                 general_nbh_coupling = "strong coupling",
                 non_spd_alerts =False
                ):
        """Construct BVAR object which follows an NIG prior, where *prior_a*, 
        *prior_b* are the priors of the inverse gamma distribution. The 
        *prior_mean_beta* and *prior_var_beta* are the hyperparameters of
        the normal prior on the regression coefficients.
        If *separate_intercepts* = True, then each spatial location gets its own
        intercept. *nbh_sequence* gives the sequence of lag-neighbourhoods 
        that are used for modelling the mean. The *padding* variable gives us
        a way to select how we want to pad the locations that are outside our 
        window.
        
        
        ###****************************************************************###
        ###                            MODEL PRIORS                        ###
        ###****************************************************************###
        Inputs always needed. They correspond to priors in the following model:
                Y ~ N(X*beta, sigma2 * I), 
                beta ~ N(beta_0, sigma2 * V_0), 
                sigma2 ~ IG(a,b)
            
        prior_a: float >0:
            a parameter of the Inverse Gamma, a>0
            
        prior_b: float >0:
            b parameter of the Inverse Gamma, b>0
            
        prior_mean_beta: 1D-numpy array of size k, k = num regressors:
            corresponds to beta_0, i.e. the mean prior of coefficients.
            Takes precedence over prior_mean_scale if both specified.
            
        prior_var_beta: 2D-numpy array of size kxk, k = num regressors:
            corresponds to V_0, i.e. the covariance prior of coefs 
            Takes precedence over prior_var_scale if both specified.
            
        prior_mean_scale: float:
            If prior_mean_beta is None, prior_mean_scale supplied, the number
            of regressors k is calculated automatically and 
            beta_0 = prior_mean_scale * np.ones(k)
        
        prior_var_scale: float >0:
            If prior_var_beta is None, prior_var_scale supplied, the number
            of regressors k is calculated automatically and 
            beta_0 = prior_var_scale * np.identity(k)
            
        
        ###****************************************************************###
        ###                 REGULAR GRID + STRONG PARAM BINDING            ###
        ###****************************************************************###
        Inputs needed when assuming regular grid with strong parameter binding:
            nbh_sequence, restriction_sequence, padding
        
        nbh_sequence: array with integer entries, only needed if data on 
                      regular grid, with strong coupling between effects:
            0, 4, 8   -> specify the sequence ofVAR-nbhs.
                         corresponds to strong parameter coupling on regular
                         grid with no neighbourhood (0), 4-neighbourhood (4), 
                         and 8-neighbourhood (8). I.e. all locations are on
                         a grid defining their nbhs, and share params.
                         (See restriction_sequence for param sharing)
        
        restriction_sequence: array with integer entries, only needed if data 
                       on regular grid, with strong coupling between effects:
            0, 4, 8 -> specify the restriction of nbh_sequence on regular
                        spatial grid with parameter coupling.
                        Regardless of 0,4,8, we always couple across all 
                        LOCATIONS! I.e., params the same across the grid.
                        However, we can vary how much we couple params within
                        each location's nbh: Not at all, i.e. one parameter
                        for each nbh location relative to each location (0), 
                        the 4 inner and the 4 outer (4), and in the case of
                        a 8-nbh, all 8 together (8). See Fig. 2 in the paper
                        for illustration of 4-nbh (red), 8 nbh (red + blue),
                        0 nbh (orange).
                        
                        NOTE: The parameter bindings for the intercepts are
                        again specified via intercept_grouping (see below).
                        They are NOT strongly coupled unless the argument
                        is not specified or is supplied as None.
        
        padding: string: 
            ONLY needed if we specify nbh_sequence and restriction_sequence,
            implying that we are on a regular grid. Then, we need to pad the
            outside of the grid using one of the below options:
            'overall_mean'  -> compute mean across space and fill in
            'row_col_mean'  -> compute row and col means and fill in
            'zero'          -> insert zeros (bias estimation towards 0)
            'leave-out'     -> don't pad at all, and estimate only using
                                locations with full neighbourhood
              
                  
        ###****************************************************************###
        ###                 GENERAL NBHS + ANY PARAM BINDING               ###
        ###****************************************************************###  
        Inputs needed when assuming general nbh structures with arbitrary 
        parameter bindings: 
            intercept_grouping, general_nbh_sequence, 
            general_nbh_restriction_sequence , general_nbh_coupling      
                      
        intercept_grouping: GxS1xS2 numpy array of ones or zeros grouping the 
            locations into G groups so that each group shares the intercept.
            Notice that summing over the G-dimension, we would get an S1xS2
            array of only ones. I.e., each location has to be in one of the G
            groups. Extreme cases: G=1 with a single slice of ones => all 
            locations have one shared intercept. G=S1*S2 with each of the G 
            slicescontaining exactly a single 1-entry and only zeros otherwise 
            => each location has individual intercept.
            
        general_nbh_sequence: list of list of lists:
            Gives an nparray of nparrays of nparrays of  
            coordinates/identifiers, i.e. an object like
            [[[2,3,4],[5,6],[7]], [[5,6],[8],[9,10]], ...].
            Here, [2,3,4] would be the 'closest' nbh to the 
            point with spatial coordinate 0, [5,6] the second-
            closest, [7] the third-closest. how far away from
            the closest nbh you consider the data is implied
            by the general_nbh_restriction_sequence that 
            will give you the indices of the nbhs to be 
            considered for each lag length.
            
            In the notation of the PAPER, this gives you the nbh. system as
            [[N_1(1), N_2(1), N_3(1)], [N_1(2), N_2(2), N_2(3)], ...], i.e. 
            list entry s belongs to location with index s and contains n neigh-
            bourhoods N_1(s), ... N_n(s) s.t. the indices describe spatial 
            closeness, with smaller indices indicating that we are closer to s.
            Note that we assume n to be the same for all locations. If there 
            is a case where you assume that some locations s have less nbhs 
            than others, simply add some empty nbhs, i.e. N_i(s) = [].
                                                          
        general_nbh_restriction_sequence: list of list:
            Gives you a list of lists of indices, i.e.
            [[0,1,2,3], [0,1,2], [0],[]], where it must hold that
            later entries are strict subsets of previous ones
            s.t. the largest value at position l is at most as
            big as the largest value at position l-1. Also, if 
            k is the largest value at position l, it must hold
            that all k' s.t. 0<=k'<=k must be in that list entry
            NOTE: If you want to have only auto-regressive 
            terms at some nbh (I.e., only its own past influen-
            ces the present), then simply add an empty list [].
            
            In the notation of the PAPER, this is the function p(.) assigning
            temporal meaning to the neighbourhood structure. p is given in 
            list form so that the l-th entry of p gives all indices that 
            are going to be used at lag length l. I.e., assuming p(l) = 3 
            (using p's definition from the paper), it will hold that the 
            respective entry in general_nbh_restriction_sequence is going to
            be [0,1,2]. More generally, for p(l) = k, [0,1,...,k-1].
            
        general_nbh_coupling: string:
            ["no coupling", "weak coupling", 
            "strong coupling"], tells you how neighbourhoods 
            are tied together. "no coupling" means that each
            linear effect of s' \in N_i(s) is modelled sepa-
            rately. "weak coupling" means that the linear 
            effect for all s' \in N_i(s) are modelled together,
            and "strong coupling" means that the linear effects
            are also modelled together across space, i.e. 
            s' \in N_i(s) and g \in N_i(k) have the same effect
            (but s' in N_j(s) and g in N_i(k) do not)
            NOTE: no coupling is not implemented, because you
            can obtain the same effect by weak coupling and
            treating each station as its own nbh.
            
            In the PAPER, notes on this are right after SSBVAR definition.
            "weak coupling" is the standard modelling framework that assumes
            that for all locations in a given nbh, we have a single linear 
            effect. "strong coupling" means that in addition, we have the same
            linear neighbourhood effect for each location.                               

        
        ###****************************************************************###
        ###                 HYPERPARAMETER LEARNING                        ###
        ###****************************************************************###  
        Inputs needed when doing hyperparameter learning: 
            hyperparameter_optimization [don't use auto_prior_update!] 
                            
        hyperparameter_optimization (ProbabilityModel level): string or None:
            -> [True, False, None, "caron", "online", "turner"]
            by default, this is True, which amounts to updating
            the gradients but not performing on-line/caron's 
            hyperpar. learning. If False or None, the gradients 
            are not updated. "caron" and "online" both employ
            the on-line hyperparameter learning proposed by
            Caron, Doucet, Gottardo (2012). If you don't want
            this, but only want to do Turner's routine, you
            have to do so via an enclosing HyperparameterOptimization 
            object. For this, put hyperparameter_optimization
            to True (default) or "turner".
            
            I.e., "turner", "True" mean that gradients are updated recursively,
            but not used (unless an enclosing HyperparameterOptimization
            object uses them), "caron" and "online" mean that we perform 
            gradient descent updates as in the PAPER. "False" and None mean
            that we don't update the gradients. (barely any computational 
            benefit in so doing)
            
        auto_prior_update: boolean.
            Basically, DON'T set to True. It updates the priors by setting them
            to the posterior expectations at time t. For instance, the beta_0
            prior at time t will be set to
                sum_r{ beta_rt[r,:] * P(r|y_1:t) }.
                
                
        ###****************************************************************###
        ###                EXOGENEOUS/ADDITIONAL PREDICTORS                ###
        ###****************************************************************###  
        NOT IMPLEMENTED!!!!  
        Inputs needed when using additional variables: 
            exo_selection, nbh_sequence_exo      
                             
                                
        NOTE: Intercepts, EXO, and ENDO vars can always ge grouped by the 
              following simple procedure: Suppose you have two groups G1, G2.
              Let's assume you assume the same model in G1, G2 but with diff-
              erent parameterizations. Lets say the params you want are
              a(G1), a(G2), b(G1), b(G2). Then you can just estimate all four
              coefs jointly by having G1 have a column of zeros for the var
              corresponding to a(G2), b(G2) and vice versa.
              
        NOTE: At some point, it may be good to replace the strings indicating
              our neighbourhood structures using integers instead, since
              string-computations are more expensive than integer-computations
              
        exo_selection:
           0,1,2,..     -> gives you a selection vector of length 
                                num_exo_vars allowing you to select which exos
                                you want to regress on Y. The integers are
                                the row index in vector [exo1, exo2, ...] of 
                                regressors available at each location. 
        
        nbh_sequence_exo:
            0,4,8           -> gives you the nbh of the lagged exos that are
                                regressors for your problem. Starts at time t
                                (rather than t-1, as for endo sequence)
                                
        ###****************************************************************###
        ###                        OTHER INPUTS                            ###
        ###****************************************************************###  
        None of these inputs are needed, they provide additional functionality
        
        non_spd_alerts: boolean:
            Gives an alert whenever the covariance matrix was not semi-positive
            definite and needed to be converted into an spd-matrix by forcing 
            it via 'nearestPD' or adding a disturbance. 
            
            NOTE: If you experience this a lot, try to rescale your data, i.e.
            normalize it on-line or do something along the same lines.
        """

        
        """STEP 1: Store priors"""
        self.a, self.b = prior_a, prior_b
        """if beta_0 or beta's covariance matrix are specified, that takes
        precedence over a supplied scaling of a vector/matrix of ones"""
        if not prior_mean_beta is None:
            self.prior_mean_beta = prior_mean_beta.flatten()
        else:
            self.prior_mean_beta= prior_mean_beta   
        if not prior_var_beta is None:
            self.prior_var_beta = prior_var_beta
        else:
            self.prior_var_beta= prior_var_beta
      
        
        """STEP 2: Store execution parameters"""
        self.auto_prior_update = auto_prior_update #Don't use
        if (hyperparameter_optimization is not None or 
            hyperparameter_optimization is not False):
            self.a_old = prior_a + 0.0000001 #Used for gradient computation
            self.b_old = prior_b+ 0.0000001 #Used for gradient computation
            self.gradient_old = 0.0         #Used for gradient computation
            self.a_list, self.b_list = [],[]
        self.hyperparameter_optimization = hyperparameter_optimization
        self.non_spd_alerts = non_spd_alerts #if cov mat not spd and forced
                                             #to be, this alerts you.
        
        """STEP 3: Get informations about the model we set up"""
        self.has_lags = True #needed inside detector   
        self.S1, self.S2 = S1, S2
        
        """STEP 3.1: If we are on a regular grid with strong param binding"""
        self.restriction_sequence = restriction_sequence
        self.nbh_sequence = nbh_sequence
        self.padding = padding
        
        """STEP 3.2: If we are on general neighbourhood structures"""
        self.general_nbh_sequence = general_nbh_sequence
        self.general_nbh_restriction_sequence = general_nbh_restriction_sequence
        self.general_nbh_coupling = general_nbh_coupling
        self.intercept_grouping = intercept_grouping
        
        #DEBUG: Add a step checking that we have either a general or regular
        #       nbh sequence, create an internal boolean/mode variable           
        
        """STEP 3.3: If we have any exogeneous/additional variables"""
        if exo_selection is None or exo_selection == []:
            self.exo_bool = False
            exo_selection = []
        else:
            self.exo_bool = True
        
        
        """STEP 4: Convert the neighbourhood into a sequence of strings
                    for the endogeneous variables"""
        endo_vars = []
        
        """STEP 4.1: Get the intercept regressor codes"""
        if (intercept_grouping is None or 
            intercept_grouping == np.array([])):
            self.intercept_codes = ["intercept"]
        elif isinstance(intercept_grouping, str) and intercept_grouping == "separate":
            self.intercept_codes = ["separate_intercepts"]
            print("NOT IMPLEMENTED YET!")
        else:
            self.num_intercept_groups = intercept_grouping.shape[0]
            self.intercept_grouping = intercept_grouping
            self.intercept_codes = []
            for g in range(0, self.num_intercept_groups):
                self.intercept_codes.append(("intercept_group_" + str(g)))
        
        #DEBUG: Step 4.2 should be outsourced into a helper function,
        #       as it clogs up space. 
        
        """STEP 4.2: Get the endogeneous regressor codes"""
        if not (restriction_sequence is None):
            """STEP 2.2A: If we use the 4-nbh sequence formulation"""
            self.lag_length = np.size(nbh_sequence)
            for lag in range(0,int(self.lag_length)):
                restriction = restriction_sequence[lag]
                nbh = nbh_sequence[lag]
                if restriction == 0:
                    if nbh == 0:
                        endo_vars.append(["center"])
                    elif nbh == 4:
                        endo_vars.append([ "center","top", "left", "right", 
                                "bottom"])
                    elif nbh == 8:
                        endo_vars.append(["center", 
                                "top", "left", "right", "bottom",
                                "topleft", "topright","bottomleft", "bottomright"])
                elif restriction == 4: 
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence") 
                        print("contained 4, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                    elif nbh == 8:
                        endo_vars.append(["center", "4_outer_nbh_res", 
                                 "4_inner_nbh_res"])
                elif restriction == 8: 
                    if nbh == 0:
                        endo_vars.append(["center"])
                        print("Warning: Restriction sequence") 
                        print("contained 8, nbh sequence a 1-nbh")
                        print("at the same position.\n")
                    elif nbh == 4:
                        endo_vars.append(["center", "4_inner_nbh_res"])
                        print("Warning: Restriction sequence") 
                        print("contained 8, nbh sequence a 4-nbh")
                        print("at the same position.\n")
                    elif nbh == 8:
                        endo_vars.append(["center", "8_nbh_res"]) 
                        print("Warning: Restriction = 8, which is not fully implemented")
#        elif general_nbh_coupling == "strong coupling":
#            """STEP 2.2B: If we use the general nbh sequence formulation"""
#            self.lag_length = int(len(general_nbh_restriction_sequence))
#            for lag in range(0, int(self.lag_length)):
#                new_endo_vars_entry = ["center"]
#                g = 0
#                #DEBUG: restriction sequence should be for each lag!
#                """g will give the index of the neighbourhoods that are part of
#                the collection of nbhs with non-zero effect at the l-th lag"""
#                for nbh_code in general_nbh_restriction_sequence[g]:
#                    new_endo_vars_entry.append("general_nbh_" + 
#                                  str(lag) + "_" + str(g))
#                    g += 1
#                endo_vars.append(new_endo_vars_entry)
                
        elif general_nbh_coupling == "weak coupling":
            """STEP 2.2C: If we use the general nbh sequence formulation with 
            weak coupling (i.e. nbh-specific, but not across space).
            Recall that the structure is as follows: 
            general_nbh_sequence = [[[4,5,6],[7,8],[9]], [[2,3,4],[5],[7]],...]
            general_nbh_restriction_sequence = [[0,1,2],[0,1,2],[0,1],[2]].
            Here, lag_length = 4, general_nbh_restriction_sequence[lag] = g(l),
            where g(l) gives you the index of the nbh generating the regressors
            at lag length l for s, i.e. N_p(l)(s)
            
            We want to get strings of form
            
            general_nbh_<lag>_<nbh_index>_<loc>, 
            
            where <lag> gives you the index in general_nbh_restriction_seq that
            you need, say <lag> = 0, i.e. we care about [0,1,2]. Given this 
            index list, <nbh_index> then tells us which of the indices (and
            thus neighbourhoods) we care about, i.e. nbh_index = 0 would mean
            we care about [0,1,2][0] = [0]. Lastly, the <loc> tells us which
            index on the lattice we care about, allowing us to retrieve 
            general_nbh_sequence[<loc>][general_nbh_restriction_seq[<lag>][<nbh_index>]]
            as the indices of the nbh with <nbh_index> corresponding to 
            <loc>'s neighbourhood at lag <lag>+1 
            """
            self.lag_length = int(len(general_nbh_restriction_sequence)) #DEBUG: 0 implies that we have res per loc
            self.empty_nbhs = [] #helps us to sort out the extraction list later
            self.sum_empty_nbhs_per_lag = np.zeros(self.lag_length)
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = []
                #g = 0
                
                #for nbh_index in relevant_nbh_indices:
                for location in range(0, S1*S2):
                    #DEBUG: This marks the center for each location separately
                    #       make sure that this does not cause problems for how
                    #       we find the lag (e.g., by counting # of "center"s)
                    new_endo_vars_entry.append("general_nbh_" + 
                          str(lag) + "_" + "center" + "_" + 
                          str(location))
                    self.empty_nbhs.append(False)
                    #DEBUG: restriction sequence should be for each lag!
                    relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]
                    
                    for nbh_index in relevant_nbh_indices:
                    #print(nbh_code)
                        """Only add the nbh if it is non-empty"""
                        if nbh_index:
                            if self.general_nbh_sequence[location][nbh_index]:
                                new_endo_vars_entry.append("general_nbh_" + 
                                  str(lag) + "_" + str(nbh_index) + "_" + 
                                  str(location))
                                self.empty_nbhs.append(False)
                            else:
                                """Mark which neighbourhoods were left out because
                                they were empty. Needed for extraction_list and
                                lag_counts"""
                                self.empty_nbhs.append(True)
                                self.sum_empty_nbhs_per_lag[lag] += 1
                        #g += 1
                    endo_vars.append(new_endo_vars_entry)
                    new_endo_vars_entry = []
            
            #DEBUG: I could now loop over all endo_vars and identify 'empty'
            #       neighbourhoods, i.e. where [] is the element in the nbh
            #       sequence. These elements could be deleted from the
            #       model, and the algo should still run as before (only 
            #       without those kicked-out values).  
                    
                    
        elif general_nbh_coupling == "strong coupling":
            """STEP 2.2D: In this case, we have the same input as for weak
            coupling, but a different interpretation. In particular, we couple
            the effects over different spatial locations. Accordingly, we save
            
            general_nbh_<lag>_<nbh_index> only.
            
            Then, in the extractors, we loop over <loc> to retrieve the 
            regressors in a single column as
            
            regressor(<lag>, <nbh_index>)[<loc>] = sum over all measurements
                at time t - <lag> for nbh given by
                gen_nbh_seq[<loc>][gen_nbh_res_seq[<lag>][<nbh]].
            """
            self.lag_length = int(len(general_nbh_restriction_sequence)) #DEBUG: 0 implies that we have res per loc
            for lag in range(0, int(self.lag_length)):
                new_endo_vars_entry = ["general_nbh_" + str(lag) + "_center"]
                #g = 0
                
                #for nbh_index in relevant_nbh_indices:
                #for location in range(0, S1*S2):
                    #DEBUG: This marks the center for each location separately
                    #       make sure that this does not cause problems for how
                    #       we find the lag (e.g., by counting # of "center"s)
                #new_endo_vars_entry.append("general_nbh_" + str(lag) + "_center")
                    #DEBUG: restriction sequence should be for each lag!
                    
                relevant_nbh_indices = self.general_nbh_restriction_sequence[lag]
                    
                for nbh_index in relevant_nbh_indices:
                    #print(nbh_code)
                
                    new_endo_vars_entry.append("general_nbh_" + 
                              str(lag) + "_" + str(nbh_index))
                        #g += 1
                endo_vars.append(new_endo_vars_entry)
                #new_endo_vars_entry = []
                    
        elif general_nbh_coupling == "no coupling":
            """STEP 2.2E: If we use the general nbh sequence formulation with 
            no coupling (i.e. nbh-specific, and for each inddividual member
            of that nbh)"""
            print("No coupling not implemented yet")
            #DEBUG: Does 'no coupling' even make sense to implement? I.e.,
            #       if we just have all general_nbh_sequence elements being
            #       singletons, we can already capture it.
        
        #self.empty_nbhs = empty_nbhs
        #self.sum_empty_nbhs_per_lag = sum_empty_nbhs_per_lag
        self.endo_vars = endo_vars


        
        """STEP 3: Define quantities relating to the exo and endo regressors:
                    the sequences of variables, the counts of variables, and
                    the lag-structure"""
                    
        """STEP 3.1: Get the regressor sequences"""
        self.exo_vars = [self.intercept_codes + exo_selection] # + all other exo vars
        self.nbh_sequence_exo = nbh_sequence_exo #DEBUG: Not used at this point
        self.exo_selection = exo_selection
        #self.exo_lag_length = len(self.nbh_sequence_exo) #DEBUG: Not used at this point

        self.all_vars = list(self.exo_vars) + list(endo_vars) 
        self.all_vars = sum(self.all_vars, [])

        """STEP 3.2: Get the number of each type of variable"""
        self.num_exo_regressors = len(sum(self.exo_vars, []))
        self.num_endo_regressors = len(sum(self.endo_vars, []))
        self.num_regressors = (self.num_endo_regressors + 
                               self.num_exo_regressors)
        
        """STEP 3.3: Get the lag structure such that lag_counts stores the
                     #exo_vars at position 0,and stores at position l the count
                     {#exo_vars + sum(#endo_vars: lag <= l)"""   
        self.lag_counts = [self.num_exo_regressors]
        last_count = self.num_exo_regressors
        if not (restriction_sequence is None):
            """STEP 3.3.A: If 0/4/8 nbhs used: Can be done via endo vars"""
            for entry in self.endo_vars:
                self.lag_counts.append(last_count + len(entry) + 1)
                last_count = last_count + len(entry) + 1 #update
        elif general_nbh_coupling == "strong coupling":
            """STEP 3.3.B: Similar to weak coupling, except you don't need to 
            multiply by the numbers of locations"""
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)))
                last_count = last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1))
        elif general_nbh_coupling == "weak coupling":
            """STEP 3.3.C: If general nbhs, we need more care"""
            """each gen_res corresponds to a lag and gives you a set at 
            position l, e.g. [0,1,2] at position 0, telling you that at the 
            first lag, the neighbourhoods used are 0,1,2. Thus, at the first 
            lag, each location has 3 regressors corresponding to the first
            three nbhs for that location in general_nbh_sequence PLUS the 
            autoregressive term, which is always incorporated but not repre-
            sented in any regressor code. 
            I.e., we need [len([0,1,2]) + 1]*S1*S2 to get the #endogeneous 
            variables at lag 1. Generally, we thus need 
            [len(gen_nbh_res_seq[l]) + 1]*S1*S2"""
            for lag in range(0, self.lag_length):
                self.lag_counts.append(last_count + (
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)
                    *self.S1*self.S2) - self.sum_empty_nbhs_per_lag[lag])
                last_count = last_count + ( - self.sum_empty_nbhs_per_lag[lag] + 
                    (len(self.general_nbh_restriction_sequence[lag]) + 1)
                    *self.S1*self.S2)
        elif general_nbh_coupling == "no coupling":
            print("No coupling not implemented yet/unneccessary since" + 
                  "subcase of weak coupling")
                    
            
        """STEP 4: Get the extraction vector and the insertion position. Note
        that the result will be a list of form [1,1,1,0,0,1,1,1,1,0,0,0], which
        means that the first 3 endogeneous variables will be kept, the next 
        two will be discarded, the next 4 will be kept, and the next 3 disc."""
        
        """STEP 4.1: see how many rows/columns you have to delete if 
                    nbh(i)>nbh(i+1). For restriction == 4, we couple together
                    the 4 adjacent regressors. For restriction == 0, we don't"""
        if not (restriction_sequence is None):
            """STEP 4.1A: If we have the 4-nbh structure"""
            multiplier = []
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                res = self.restriction_sequence[l]
                nbh = self.nbh_sequence[l]
                if res == 0:
                    multiplier.append(1)
                    endo_regressors_per_lag.append(int(nbh) + 1)
                elif res == 4:
                    multiplier.append(0.25)
                    endo_regressors_per_lag.append(int(nbh*0.25) + 1)
            multiplier = np.array(multiplier)
        else:
            """STEP 4.1B: If we have a general nbh structure, we get 
            endo_regressors_per_lag differently. In particular, just look at
            the self.endo_vars object."""
            #DEBUG: False. ALso: NOT used!
            endo_regressors_per_lag = []
            for l in range(0, self.lag_length):
                endo_regressors_per_lag.append(int(len(self.endo_vars[l])))
            
                
        """STEP 4.2: Given the multiplier, you can now get a list that tells
                     you for given X_t-1 which columns need copying to X_t.
                     You never copy exogeneous variables. Also, the first lag 
                     for X_t will be new, so one can copy at most lag_length -1
                     neighbourhoods from X_t-1 to X_t.
                     NOTE: Extraction list is used on the detector-level, so 
                           regressors passed to BVAR_NIG objects are already
                           the right ones"""
                           
                           
        """"STEP 4.2: Get the list of variables to be extracted from X_t-1
                        by looking at the endo regressors needed per lag"""
        #DEBUG: Incorrect if we have weak coupling!
        self.extraction_list = [False]*(self.num_exo_regressors) 
        
        if not (restriction_sequence is None):
            """STEP 4.2 A: IF we have 0/4/8 nbhs """
            for i in range(0,self.lag_length-1):
                self.extraction_list = (self.extraction_list 
                    + [True]*endo_regressors_per_lag[i+1]
                    + [False]*int(endo_regressors_per_lag[i] - 
                                          endo_regressors_per_lag[i+1]))
            #if (i != self.lag_length-2):
               # self.extraction_list_endo = (self.extraction_list_endo
                #               + [False]*int(endo_regressors_per_lag[i] - 
                #                          endo_regressors_per_lag[i+1]))
                               #+ [True]*int(self.nbh_sequence[i+1]*multiplier) 
                               #+ [False]*int((self.nbh_sequence[i] - 
                               #  nbh_sequence[i+1])*multiplier))
            """STEP 4.3 A: The last lag of X_t-1 will 'slide out' of sight, so it 
                       definitely is not needed for X_t anymore."""
            self.extraction_list += ([False]*
                                endo_regressors_per_lag[self.lag_length-1])

            
        elif general_nbh_coupling == "weak coupling":
            """STEP 4.2 B: IF we have general nbhs"""
            #DEBUG: We need to adapt this for the weak nbh modification!
            #       since we now allow that some locations don't have 
            #       nbhs (i.e. empty lists dont have a regressor code),
            #       the number of params is not just S1*S2*len(gen_nbh_res).
            #       Now we cannot just repeat it over and over.
            #NOTE:  If we have a [] element, it means that 
            per_location = []
            for lag in range(0, self.lag_length-1):
                num_retained = (1 + len(np.intersect1d(
                            self.general_nbh_restriction_sequence[lag],
                            self.general_nbh_restriction_sequence[lag+1])))
                num_discarded = ( -num_retained + 1 + 
                        len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True]* num_retained + 
                                 [False] * num_discarded)
            """STEP 4.3 B: The last lag of X_t-1 will 'slide out' of sight, so it 
                       definitely is not needed for X_t anymore."""
            total_num_last_lag = 1+ len(
                    self.general_nbh_restriction_sequence[self.lag_length-1])
            per_location += ([False]* total_num_last_lag)
            
            """STEP 4.4 B: Use that we have the same structure all across the 
            lattice, and simply multiply each entry of 'per_location' by the
            number of lattice elements"""
            self.extraction_list += sum(
                [self.S1*self.S2*[e] for e in per_location],[])
            self.extraction_list[self.num_exo_regressors:] = np.array(
                self.extraction_list)[np.where(np.array(
                        self.empty_nbhs) == False)].tolist()
            #print(self.extraction_list)
            #self.extraction_list = self.extraction_list[ 
            #        [not i for i in empty_nbhs]]
            
        elif general_nbh_coupling == "strong coupling":
            """STEP 4.2 B: IF we have general nbhs"""
            per_location = []
            for lag in range(0, self.lag_length-1):
                num_retained = (1 + len(np.intersect1d(
                            self.general_nbh_restriction_sequence[lag],
                            self.general_nbh_restriction_sequence[lag+1])))
                num_discarded = ( -num_retained + 1 + 
                        len(self.general_nbh_restriction_sequence[lag]))
                per_location += ([True]* num_retained + 
                                 [False] * num_discarded)
            """STEP 4.3 B: The last lag of X_t-1 will 'slide out' of sight, so it 
                       definitely is not needed for X_t anymore."""
            total_num_last_lag = 1+ len(
                    self.general_nbh_restriction_sequence[self.lag_length-1])
            per_location += ([False]* total_num_last_lag)
            
            """STEP 4.4 B: Use that we have the same structure all across the 
            lattice, and simply multiply each entry of 'per_location' by the
            number of lattice elements"""
            self.extraction_list += per_location
        
        """STEP 4.5: In order to copy entries of X_t-1 to X_t, you need to know
                     the position of X_t at which you should insert. (This does
                     only affect the endogeneous part of the regressors)"""
        self.insertion_position = - sum(self.extraction_list)
                                    #(self.num_exo_regressors 
                                  # + endo_regressors_per_lag[0] - 1)
                               
        """STEP 5: create the objects we need to trace through time"""
        self.XX = None
        self.YX = None
        #self.joint_log_probabilities = None 
        self.model_log_evidence = -np.inf
        
        #DEBUG: These are not really needed, but due to how the detector calls
        #       model_and_run_length_distr, they are needed for the time before
        #       initialization takes place.
        self.retained_run_lengths = np.array([0,0])
        self.joint_log_probabilities = 1
        
        #print(prior_var_beta)
        """STEP 6: Rectify prior_beta_mean and prior_beta_var if needed"""
        if ((self.prior_mean_beta is None) or (self.prior_var_beta is None) or
            (self.num_regressors != np.size(self.prior_mean_beta)) or
            (self.num_regressors != prior_var_beta.shape[0])):
            if prior_mean_scale is None:
                prior_mean_scale = 0
            if prior_var_scale is None:
                prior_var_scale = 100
            self.prior_mean_beta = prior_mean_scale*np.ones(self.num_regressors)
            self.prior_var_beta = (prior_var_scale*
                                   np.identity(self.num_regressors))
            
            
    def reinstantiate(self, a = None, b = None):
        """Return a new BVARNIG-model that contains all the same attributes as
        this BVARNIG model. In some sense, it is an 'emptied' version of the
        same model"""
        
        """STEP 1: Get all the characteristics of this model"""
        prior_a, prior_b, S1, S2 = self.a, self.b, self.S1, self.S2 
        prior_mean_beta,prior_var_beta=self.prior_mean_beta,self.prior_var_beta
        nbh_sequence = self.nbh_sequence
        restriction_sequence = self.restriction_sequence
        intercept_grouping = self.intercept_grouping
        general_nbh_sequence = self.general_nbh_sequence
        general_nbh_restriction_sequence = self.general_nbh_restriction_sequence
        nbh_sequence_exo = self.nbh_sequence_exo
        exo_selection = self.exo_selection
        padding = self.padding
        auto_prior_update = self.auto_prior_update
        hyperparameter_optimization = self.hyperparameter_optimization
        general_nbh_coupling = self.general_nbh_coupling
        non_spd_alerts = self.non_spd_alerts
        
        """STEP 2: Check whether you have a new prior already"""
        if a is None:
            a = prior_a
        if b is None:
            b = prior_b
        
        """STEP 2: Use the characteristics to clone the model"""
        clone_model = BVARNIG(prior_a = a, prior_b = b, S1=S1, S2=S2,
              prior_mean_beta=prior_mean_beta,
              prior_var_beta =prior_var_beta,
              prior_mean_scale=None, prior_var_scale=None,
              nbh_sequence=nbh_sequence, 
              restriction_sequence=restriction_sequence,
              intercept_grouping=intercept_grouping, 
              general_nbh_sequence=general_nbh_sequence,
              general_nbh_restriction_sequence=general_nbh_restriction_sequence,
              nbh_sequence_exo=nbh_sequence_exo, exo_selection=exo_selection,
              padding=padding, auto_prior_update=auto_prior_update, 
              hyperparameter_optimization=hyperparameter_optimization,
              general_nbh_coupling=general_nbh_coupling,
              non_spd_alerts=non_spd_alerts)
        
        """STEP 3: Return the cloned model"""
        return clone_model
    
    def get_hyperparameters(self):
        """Simply returns a and b, used by HyperparameterOptimization"""
        return [self.a, self.b]
        
        
    #DEBUG: Check that x_exo has 3 dimensions (rather than 4), or code exo_selection
    #       differently inside the detector object
    #NOTE: We need to pass X_endo with one more entry into this function,
    #       namely for y_2!
    def initialization(self, X_endo, X_exo, Y_2, X_exo_2, cp_model, model_prior,
            padding_columns_computeXX = None, padding_column_get_x_new = None):
        """Initialize the model (i.e. t=1) with some inputs from the 
        containing Detector object. The padding_column arguments are only 
        needed for the demo Csurf object.
        
        NOTE:    The exo_selection list is applied inside detector, so X_exo
                    will already contain everything relevant
        """
        print("Initializing BVAR object")
        
        """STEP 0: Get relevant variables and reshape them"""
        Y1 = X_endo[-1,:].flatten() 
        Y2 = Y_2.flatten()
        X1_endo = X_endo[:self.lag_length,:].reshape(self.lag_length, 
                        self.S1, self.S2)

        """In case there are no exogeneous variables in this model, take 
        the relevant precautions."""
        if self.exo_bool:
            #DEBUG: RESHAPE will not corr. to real dims of exo vars
            X1_exo = (X_exo[-1,:,:].reshape(
                self.num_exo_regressors, self.S1, self.S2))
        else:
            X1_exo = None
        
        """STEP 1: Format quantities correctly & compute them using 
        neighbourhood structure"""
        
        """STEP 1.1: Quantities for time point t"""
        self.XX = np.zeros(shape=(self.num_regressors,self.num_regressors))
        self.XY = np.zeros(self.num_regressors)
        self.X_t = np.zeros(shape=(self.S1*self.S2, self.num_regressors))
        self.X_tp1 = np.zeros(shape=(self.S1*self.S2, self.num_regressors))
        self.YY = np.inner(Y1, Y1)
        
        """STEP 1.2: Quantities for time point t and run-length r"""
        self.XX_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors)) #2 for r=-1 and r=0
        self.XY_rt = np.zeros(shape=(2,self.num_regressors))  #2 for r=-1 and r=0
        self.YY_rt = np.zeros(2)
        
        #DEBUG: QR
        self.Q_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors))
        self.R_rt = np.zeros(shape=(2,self.num_regressors, self.num_regressors))
        
        #self.small_matrices_inv = np.zeros(shape=(2, self.S1*self.S2, 
        #                                          self.S1*self.S2))
        self.M_inv_1_rt = np.zeros(shape=(2,self.num_regressors, 
                                          self.num_regressors))
        self.M_inv_2_rt = np.zeros(shape=(2,self.num_regressors, 
                                          self.num_regressors))
        self.log_det_1_rt = np.zeros(2)    
        self.log_det_2_rt = np.zeros(2)
        self.beta_XX_beta_rt = np.zeros(2)
        self.beta_rt = np.zeros(shape=(2,self.num_regressors))
        
        #DEBUG: Maybe I need 1,1?
        self.retained_run_lengths = np.array([0,0])
        
        """STEP 2: Compute prior- and data-dependent quantities:
                    Computation of X, X'X,  X'Y, and Y'Y from scratch."""
        self.compute_X_XX_XY_YY( Y1, X1_endo, X1_exo, padding_columns_computeXX) # gives us X, X'X, X'Y, Y'Y
        #DEBUG: Somehow we lose a dimension.
        self.X_tp1 = self.get_x_new(Y2, X_exo_2 ,1,padding_column_get_x_new) #gives us X_t+1                
                
        
        
        """STEP 3: Get the computationally burdensome quantities that are 
                   based on determinants, QR decomposition, and inverses"""
                   
        """STEP 3.1: Computation of QR decomposition QR = X'X + D^-1  
                     from scratch"""
        #DEBUG: Check that prior_var_beta has the right dimension, i.e.
        #       it is num_regressors x num_regressors
        self.D_inv = np.linalg.inv(self.prior_var_beta) #not efficient if D diagonal
        self.D_inv_Q, self.D_inv_R = np.linalg.qr(self.D_inv)
        self.D_inv_log_det =  np.sum(np.log(np.abs(np.diagonal(self.D_inv_R))))
        
        """STEP 3.1A: Loop until all of X_t has been processed. Increments of
                      X_t can at most be of size num_regressors"""
                      
        #DEBUG: QR
        Q0, R0 = self.QR_loop(self.D_inv_Q, self.D_inv_R, self.X_t)
        
        M_inv_1 = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt[0,:,:] =  self.M_inv_1_rt[1,:,:] = M_inv_1
            
        """STEP 3.1B: Fill in the results into Q_rt, R_rt"""
        self.Q_rt[0,:,:] = self.Q_rt[1,:,:] = Q0
        self.R_rt[0,:,:] = self.R_rt[1,:,:] = R0
        
        """STEP 3.2 Compute D^-1*beta_prior and beta_prior * D^-1 * beta_prior
                     which are needed later in the estimation"""
        self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
        self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)
        
        """STEP 3.3: Get the first two values of X'X_rt and X'Y_rt.
                     NOTE: Since we will only need X'Y for computing beta(r,t),
                     we need to work with (D^-1 * beta_0 + X'Y), which is why
                     we add D^-1 * beta_0 to X'Y whenever we are at r=0."""
        self.XX_rt[0,:,:] = self.XX_rt[1,:,:] = self.XX + self.D_inv
        self.XY_rt[0,:] = self.XY_rt[1,:] = (self.XY + self.D_inv_b0)
        self.YY_rt[0] = self.YY_rt[1] = self.YY
        
        
        """STEP 3.4: Use the trace to compute the determinants of Q(r,t)R(r,t)
                     for all run-lengths. These are needed in posterior of Y
                     They can be obtained as trace of R[r,:,:] because Q is an 
                     orthogonal matrix, so det(Q) = 1 and as 
                     det(QR) = det(Q)det(R), it follows det(QR) = det(R)"""
        
        #DEBUG: QR
        #diag = np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))
        #self.log_det_1_rt = np.sum(np.log(diag), axis=1)
        sign, value = np.linalg.slogdet(self.M_inv_1_rt[0,:,:])
        self.log_det_1_rt[0] = self.log_det_1_rt[1] = (value) #s.p.d. matrices have pos dets

        
        """STEP 3.5: Computation of beta = MX'Y from scratch, using triangular 
                     matrix solver! Also compute beta^T X'X(r,t) beta"""
        #DEBUG: QR
        #beta = linalg.solve_triangular(a = self.R_rt[0,:,:],
        #        b = np.matmul(np.transpose(self.Q_rt[0,:,:]),self.XY_rt[0,:]), 
        #        check_finite=False)
        beta = np.matmul(self.M_inv_1_rt[0,:,:],self.XY_rt[0,:])
        self.beta_rt[0,:] = self.beta_rt[1,:] = beta
        
        #DEBUG: DO I need to insert stuff into this array later on?
        self.beta_XX_beta_rt[0] = self.beta_XX_beta_rt[1] = (np.inner(np.matmul(
                self.beta_rt[0,:], self.XX_rt[0,:]), self.beta_rt[0,:]))
        
        
        """STEP 4: Compute the joint log probs"""
        #DEBUG: QR
        #self.M_inv_1_rt[0,:,:] = self.M_inv_1_rt[1,:,:] = linalg.solve_triangular(
        #        a=R0, b = np.transpose(Q0), check_finite=False)
        a_ = self.a + 0.5
        b_ = self.b + 0.5*(self.b0_D_inv_b0 + self.YY - self.beta_XX_beta_rt[0])
        #DEBUG: Check formula
        
        C_0_inv = (a_/b_)*(np.identity(self.S1*self.S2) - 
            np.matmul(self.X_t, np.matmul(self.M_inv_1_rt[0,:,:], 
            np.transpose(self.X_t))))
        #
        #log_det = -((self.S1*self.S2)* (np.log(b_) - np.log(a_)) + 
        #            self.D_inv_log_det - self.log_det_1_rt[0])
        #DEBUG: If b_ < 0 (i.e., the log produces a nan), we compute the 
        #       log determinant by setting b_ = eps, where eps > 0
        if b_<0:
            #b_ = 0.000001
            log_det = np.nan
        else:
            log_det = ((self.S1*self.S2) * (np.log(b_) - np.log(a_)) +
                       self.D_inv_log_det - self.log_det_1_rt[0])
        resid = Y1 - np.matmul(self.X_t, self.beta_rt[0,:]) 
        self.model_log_evidence = ( np.log(model_prior) + 
                BVARNIG.mvt_log_density(resid, C_0_inv, log_det, 2*a_, 
                                        self.non_spd_alerts))
        
        """STEP 5.4: Respect the model prior and get the joint log probs"""
        
        """Ensure that we do not get np.log(0)=np.inf by perturbation"""
        if cp_model.pmf_0(1) == 0:
            epsilon = 0.000000000001
        else:   
            epsilon = 0
        """get log-probs for r_1=0 or r_1>0. Typically, we assume that the 
        first observation corresponds to a CP (i.e. P(r_1 = 0) = 1),
        but this need not be the case in general."""
        r_equal_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(0) + epsilon)) 
        r_larger_0 = (self.model_log_evidence + 
                     np.log(cp_model.pmf_0(1)+ epsilon))   
        self.joint_log_probabilities = np.array([r_equal_0, r_larger_0]) 
        
        """Get the derivative of the log probs, too, just initialize to 1 
        (since log(1) = 0), initialize with 2 columns (for 2 hyperparams)"""
        self.model_specific_joint_log_probabilities_derivative = np.ones((2,2))
        self.model_specific_joint_log_probabilities_derivative_sign = np.ones(
                (2,2))
        #print("self.joint_log_probabilities at initialization: ",
        #      self.joint_log_probabilities)
        
        """STEP 4: Perform QR update w.r.t. X_tp1 and get M_inv + log_det_2 
                    NOTE: Do NOT update X'X, X'Y, X_t, X_tp1, Y'Y since
                          they will be already updated"""
        #DEBUG: QR
        Q1, R1 = self.QR_loop(self.Q_rt[0,:,:], self.R_rt[0,:,:], self.X_tp1)
        self.Q_rt[0,:,:] = self.Q_rt[1,:,:] = Q1
        self.R_rt[0,:,:] = self.R_rt[1,:,:] = R1
        
        #DEBUG: QR
        #self.M_inv_2_rt[0,:,:] = self.M_inv_2_rt[1,:,:] = linalg.solve_triangular(
        #        a=R1, b = np.transpose(Q1), check_finite=False)
        
        """Brute-force inversion of the small matrix"""
        small_matrix_inv = (
                np.linalg.inv(
                np.identity(self.S1*self.S2) +  
                np.matmul((self.X_tp1), np.matmul(
                        self.M_inv_1_rt[0,:,:], np.transpose(self.X_tp1)))) )
                #self.X_tp1.T.dot(self.M_inv_1_rt[0,:,:].dot(self.X_tp1)) ) )
        
        """Brute force determinant calc for small matrix + recursive update for
        determinant of M(r,t). We take -value because log(det(M^-1)) =
        -log(det(M))"""
        sign2, value2 = np.linalg.slogdet(small_matrix_inv)
        self.log_det_2_rt[0] = self.log_det_2_rt[1] = (
                value2 + self.log_det_1_rt[0])
        
        """Brute force multiplication of M(r,t)^-1 with X_tp1"""
        M_inv_1_x_X_tp1 = np.matmul(self.M_inv_1_rt[0,:,:], 
                                    np.transpose(self.X_tp1))
        """Woodbury Inversion formula for M_inv_2"""
        self.M_inv_2_rt[0,:,:] = self.M_inv_2_rt[1,:,:] = (
                self.M_inv_1_rt[0,:,:] - np.matmul((M_inv_1_x_X_tp1), 
                np.matmul( small_matrix_inv, 
                          np.transpose(M_inv_1_x_X_tp1))))
        
        #DEBUG: QR
        #diag = np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))
        #self.log_det_2_rt = np.sum(np.log(diag), axis=1)
        
        #DEBUG: Maybe add the run-lengths in here somehow?


    def evaluate_predictive_log_distribution(self, y, t):
        """Returns the log densities of *y* using the predictive posteriors
        for all possible run-lengths r=0,1,...,t-1,>t-1 as currently stored by 
        virtue of the sufficient statistics.             
        The corresponding density is computed for all run-lengths and
        returned in a np array
        
        Note: This is called BEFORE update_log_distr, so at time t, the 
                quantities one tracks through time (like Q_rt, R_rt, ...) will
                still only hold Q(r,t), R(r, t), ... and so on (rather than
                Q(r+1,t+1), R(r+1,t+1) ... ). Similarly, the regressors  
                X_t will actually correspond to time point t-1, so we instead 
                use the regressors stored inside X_tp1 = X_t+1 for evaluating
                the pred. density of y.
        """
        
        """STEP 0: Ensure that y has desired format"""
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)
        
        """Note that we store the r0-C_t_inv too, so this quantity is one
        entry longer than all other quantities"""
        self.C_t_inv = np.zeros((run_length_num+1, self.S1*self.S2, 
                                 self.S1*self.S2))
        self.C_t_inv[0,:,:] = self.C_t_inv_r0

        """Note: We cannot use retained_run_lengths to loop directly, since
                 r=t-1 and r>t-1 both have a 0 in there."""      
        for r in range(0,run_length_num):
            
            """STEP 1A: Get inverse using stored quantities & woodbury"""
            a_ = self.a + (self.retained_run_lengths[r]+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] - 
                            self.beta_XX_beta_rt[r]))
            #DEBUG: Check formula...
            #DEBUG: Store C_t_inv (but without the a/b part) for use in
            #       optimization routine
#            C_t_inv = (a_/b_)*(np.identity(self.S1*self.S2) - 
#                       np.matmul(self.X_tp1, np.matmul(self.M_inv_2_rt[r,:,:], 
#                       np.transpose(self.X_tp1))))
            
            self.C_t_inv[r+1,:,:] = (np.identity(self.S1*self.S2) - 
                       np.matmul(self.X_tp1, np.matmul(self.M_inv_2_rt[r,:,:], 
                       np.transpose(self.X_tp1))))
        
            """STEP 1B: Get the log determinant using the Woodbury Formula and
                     applying the determinant lemma afterwards
                     
                     NOTE: We take the minus in front because we compute the 
                           log det of the INVERSE matrix C(r,t)^-1 here, but  
                           need that of C(r,t) for the function call"""
            #DEBUG: check if b_ < 0
            #if(b_ < 0):
            #    print("b_: ", b_)
            #    print("t: ", t)
            #    print("r : ", r)
            #DEBUG: Make sure that log_det is not nan!
            if b_ < 0:
                #b_ = 0.000001
                log_det = np.nan
            else:
                log_det = ((self.S1 * self.S2)*(np.log(b_) - np.log(a_)) + 
                           self.log_det_1_rt[r] - self.log_det_2_rt[r])


            """STEP 1C: Evaluate the predictive probability"""
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r,:]) 
            #self.resid = resid
            #self.X_tp1_resid = self.X_tp1.copy()
            #self.beta_resid = self.beta_rt[r,:].copy()
            

            log_densities[r] = (
                    BVARNIG.mvt_log_density(resid, (a_/b_)*self.C_t_inv[r+1,:,:], 
                        log_det, 2*a_, self.non_spd_alerts))
            
        
        """STEP 2: return the full log density vector"""
        return log_densities
    

    def evaluate_log_prior_predictive(self, y, t):
        """use only the prior specs of BVARNIG object to get predictive prob"""
        resid = y - np.matmul(self.X_tp1, self.prior_mean_beta)
        self.C_t_inv_r0 = (
                np.identity(self.S1*self.S2) - 
                np.matmul(self.X_tp1, np.matmul(self.prior_var_beta, 
                       np.transpose(self.X_tp1))))
        _, log_det = np.linalg.slogdet((self.a/self.b)*self.C_t_inv_r0)
        #Check if log det admissible/if C_t_inv is spd?
        return min(0.0, BVARNIG.mvt_log_density(resid, 
                (self.a/self.b)*self.C_t_inv_r0, log_det, 2*self.a, True))
    
    
    def save_NLL_fixed_pars(self, y,t):
        """Similar to eval_pred_log_distr, but evaluates normal instead"""
        """STEP 0: Ensure that y has desired format"""
        y = y.flatten()
        run_length_num = self.retained_run_lengths.shape[0]
        log_densities = -np.inf * np.ones(shape=run_length_num)

        """Note: We cannot use retained_run_lengths to loop directly, since
                 r=t-1 and r>t-1 both have a 0 in there."""      
        for r in range(0,run_length_num):
            
            """STEP 1A: Get inverse using stored quantities & woodbury"""
            a_ = self.a + (self.retained_run_lengths[r]+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] - 
                            self.beta_XX_beta_rt[r]))
            sigma2 = max((b_/(a_+1)), 0.0000001)
            cov_mat = sigma2*self.C_t_inv[r+1,:,:]

            """STEP 1C: Evaluate the predictive probability"""
            resid = y - np.matmul(self.X_tp1, self.beta_rt[r,:]) 
            
            """normal evaluation"""
            log_densities[r] = (
                    stats.multivariate_normal.logpdf(resid, cov=cov_mat))          
        
        """STEP 2: return the full log density vector"""
        self.one_step_ahead_predictive_log_probs_fixed_pars = log_densities
        #return log_densities
        
           
    
    #DEBUG: Include X_t+1 in update, for now assume that one KNOWS the exos 
    #       i.e. at time t, we can assign X_t<=X_t+1 and compute the new X_t+1 from scratch
    #DEBUG: We will need to pass y2 into this function, too. (To compute X_t+1)
    def update_predictive_distributions(self, y_t, y_tm1, x_exo_t, x_exo_tp1, t, 
                                        padding_column_tm1 = None,
                                        padding_column_t = None, 
                                        r_evaluations = None):
        """Takes the next observation, *y*, at time *t* and updates the
        sufficient statistics, means & vars corresponding to all potential 
        run-lengths r=0,1,...,t-1,>t-1.
        """
        
        """STEP 0: Flatten y and check if x_exo needs processing"""
        y_t, y_tm1 =y_t.flatten(), y_tm1.flatten()
        
        """STEP 1: Extract the NEW regressor vectors, and do two things:
                    (A) Store them in X for the S1*S2 rank-1-updates later
                    (B) Immediately compute the new dot products for X'X, and 
                        copy the old dot products that can be re-used
                    (C) Immediately copute the new dot products for X'Y
                    (D) Update  X'X(r-1,t-1) => X'X(r,t),
                                X'Y(r-1,t-1) => X'Y(r,t),
                                Y'Y(r-1,t-1) => Y'Y(r,t) using addition"""  
        #DEBUG: we need to pass 3-dim y_t object to get_x_new, so either shape
        #       into S1xS2 or do inside get_x_new. 
        #DEBUG: Similiar for cross_prod_upd and similar for x_exo
        #PROBLEM: Somehow, X_t becomes X_tp1. Unclear why.
        #DEBUG: So we can check stuff
        self.XX_old, self.XY_old, self.X_t_old = (self.XX.copy(),
                        self.XY.copy(), self.X_t.copy())
        self.Y_new, self.Y_old = y_t, y_tm1
        
        self.regressor_cross_product_updates(y_t,  y_tm1, x_exo_t, 
                                             t, padding_column_tm1)
        #DEBUG: should be the same
        #self.X_t = self.X_tp1.copy()
        self.X_tp1 = self.get_x_new(y_t, x_exo_tp1, t, padding_column_t) #doesn't work!
                   
        """STEP 2: Update quantities that are direct functions of the data"""
        
        """STEP 2.1: Extend the run-lengths and add run-length 0, add QR for
        the new run-length r=0, and update beta and beta^T M beta. Also copy
        the entries of log_det_2_rt into log_det_1_rt to update log_det_1_rt.
        
        UPDATES:    run-lengths
                    #QR(0,t)
                    M_1_inv(0,t) added, rest of M_1_inv copied
                    beta_rt
                    beta_XX_beta
                    log_det_1_rt(0,t) added, rest of log_det_2_rt copied
                    
        """
        #DEBUG: QR
        #self.pre_QR_updates(t)
        self.pre_updates(t)

        """STEP 2.2: Bottleneck: Update your QR decomposition with the 
        regressors in X_tp1 from QR(r,t) to QR(r+1,t+1)
        
        UPDATES:    #QR(r+1, t+1)
                    M_2_inv(r+1,t+1) for all r 
                    log_det_2(r+1,t+1) for all r
        """
        #DEBUG: QR
        #self.QR_updates(t)
        self.updates(t)
        
        """STEP 2.3: Update log_det_2_rt using newly computed QRs, and 
        get the inverses M_inv_rt using those newly computed QRs
        
        UPDATES:    log_det_2_rt
                    M_inv_rt
        """
        #DEBUG: QR
        #self.post_QR_updates(t)


    def pre_updates(self, t):
        """Extend run-lengths, add QR for r=0, update beta and beta * M * beta,
        update log_det_1_rt"""
        
        """STEP 1: extend the retained run lengths"""
        self.retained_run_lengths =  self.retained_run_lengths + 1 
        self.retained_run_lengths = np.insert(self.retained_run_lengths, 0, 0)
        
        
        #DEBUG: QR
        #"""STEP 2: add the QR for r=0"""
        #newQ, newR = self.QR_loop( self.D_inv_Q, self.D_inv_R, self.X_t)
        #self.Q_rt = np.insert(self.Q_rt, 0, newQ, axis=0)
        #self.R_rt = np.insert(self.R_rt, 0, newR, axis=0)
        """STEP 2: Add the new M inverse"""
        new_M_inv = np.linalg.inv(self.D_inv + self.XX)
        self.M_inv_1_rt = np.insert(self.M_inv_2_rt.copy(), 0, new_M_inv, axis=0)
        
        """STEP 3: update the beta estimates and beta * XX * beta"""
        self.compute_betas(t)
        
        """STEP 4: update the log determinant 1 and M_inv_1_rt. Again, take
        -new_log_det because we take the log-det of the inverse of the matrix
        whose log det we wanna store."""
        #DEBUG: QR
        #new_log_det = np.sum(np.log(np.abs(np.diagonal(newR))))
        sign, new_log_det = np.linalg.slogdet(new_M_inv) 
        #self.log_det_1_rt = np.insert(self.log_det_2_rt.copy(), 0, -new_log_det)
        self.log_det_1_rt = np.insert(self.log_det_2_rt.copy(), 0, new_log_det)
        
        #DEBUG: QR
        #new_M_inv = linalg.solve_triangular(a=newR, b = np.transpose(newQ), 
        #                                    check_finite=False)
        #self.M_inv_1_rt = np.insert(self.M_inv_2_rt.copy(), 0, new_M_inv, 
        #                            axis=0)
        #DEBUG: Should be the same as computing
        #       np.sum(np.log(np.abs(np.diagonal(self.R_rt, axis1=1, axis2=2))), axis=1)
        
    
    def updates(self, t):
        """update the QR-decomposition of X'X(r, t) to X'X(r+1,t+1) via an 
        S1*S2-rank update, i.e. have decomposition for t+1 at time t
        """
        run_length_num = self.retained_run_lengths.shape[0]
        
        self.M_inv_2_rt = np.zeros((run_length_num, self.num_regressors, 
                                    self.num_regressors))#np.insert(self.M_inv_2_rt, 0, 
            #np.zeros((self.num_regressors,self.num_regressors) ),
            #axis=0)
        #DEBUG: Not sure this is correct
        self.log_det_2_rt = np.zeros(run_length_num)#np.insert(self.log_det_2_rt, 0, 0)
        
        for r in range(0,run_length_num):
            #DEBUG: QR
            #self.Q_rt[r,:,:], self.R_rt[r,:,:] = self.QR_loop(
            #        self.Q_rt[r,:,:], self.R_rt[r,:,:], self.X_tp1)
            """Compute small_matrix for r, and get inverse + det"""
            M_inv_x_X_tp1 = np.matmul(self.M_inv_1_rt[r,:,:], np.transpose(self.X_tp1))
            small_matrix_inv = np.linalg.inv(np.identity(self.S1*self.S2) + 
                    np.matmul( np.transpose(M_inv_x_X_tp1), 
                              np.transpose(self.X_tp1)))
            
            """Update M_2_inv"""
            self.M_inv_2_rt[r,:,:] = self.M_inv_1_rt[r,:,:] - np.matmul(
                    (M_inv_x_X_tp1), np.matmul(small_matrix_inv, 
                    np.transpose(M_inv_x_X_tp1) ))
            
            """Update log_det_2"""
            sign, value = np.linalg.slogdet(small_matrix_inv)
            self.log_det_2_rt[r] = value + self.log_det_1_rt[r]


    def post_QR_updates(self, t):
        """After the QR updates have been updated to contain the X_tp1 r
        regressors, obtain the M_inv_rt matrices and update log_det_2"""
        self.M_inv_2_rt = np.insert(self.M_inv_2_rt, 0, 
                        np.zeros((self.num_regressors,self.num_regressors) ),
                        axis=0)
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0,run_length_num):
            self.M_inv_2_rt[r,:,:] = linalg.solve_triangular(a=self.R_rt[r,:,:], 
                b = np.transpose(self.Q_rt[r,:,:]), check_finite=False)
        self.log_det_2_rt = np.sum(np.log(np.abs(np.diagonal(self.R_rt, 
                                            axis1=1, axis2=2))), axis=1)
        
        
    def QR_loop(self,Q0, R0, X):
        """Taking Q0, R0 as starting decomposition, this function loops over
        the elements in X_t until all row vectors have been used for rank-1
        updates. Overwrites Q0,R0 in the process."""
        current_count = end_point = 0
        while (end_point != self.S1*self.S2):
            start_point = current_count*self.num_regressors
            end_point = min((current_count+1)*self.num_regressors, 
                            self.S1*self.S2)
            current_range = range(start_point, end_point)
            Q0, R0 = linalg.qr_update( Q0, R0,
                    np.transpose( X[current_range,:]), 
                    np.transpose( X[current_range,:]), 
                    check_finite=False)
            current_count = current_count + 1
        return Q0, R0
  
    
    def compute_betas(self,t):
        """compute beta = MX'Y for all run-lengths using triangular solver:
        Since M^-1*beta = X'Y, and since M^-1 = QR with R a triangular
        matrix and Q^TQ = I, it holds that QR*beta = X'Y, and so
        R*beta = Q^T*X'Y can be solved for beta using a triangular solver
        once Q^T*X'Y has been computed (which is O(n^2)). Thus, since the
        triangular solver itself is O(n^2), we obtain beta in O(n^2)!"""
        run_length_num = self.retained_run_lengths.shape[0]
        self.beta_rt = (
            np.insert(self.beta_rt , 0, np.zeros(self.num_regressors), axis=0))
        self.beta_XX_beta_rt = np.insert(self.beta_XX_beta_rt , 0, 0, axis=0)
        for r in range(0,run_length_num):
            #DEBUG: QR
            #self.beta_rt[r,:] = linalg.solve_triangular(a = self.R_rt[r,:,:],
            #    b = np.matmul(np.transpose(self.Q_rt[r,:,:]), self.XY_rt[r,:]),
            #    check_finite=False)
            self.beta_rt[r,:] = np.matmul(self.M_inv_1_rt[r,:,:], self.XY_rt[r,:])
            self.beta_XX_beta_rt[r] = np.inner(self.beta_rt[r,:], 
                np.matmul(self.XX_rt[r,:,:],self.beta_rt[r,:]))
    
    
    def regressor_cross_product_updates(self, y_t, y_tm1, x_exo, t, 
                                        padding_column=None):
        """Get the new regressors, i.e. transform the shape of X and X'X in 
        accordance with your new observations at time t. Also update X'Y 
        and Y'Y, since it is not much more work once the other stuff is 
        in place. The argument *padding_column* is only needed for the demo
        object BVAR_NIG_CSurf for the column that is next to the C Surface."""
        
        """STEP 1: Copy all old regressors from  X and X'X_t into new X and 
        X'X and shift them within the same data structure"""
        
        """STEP 1.1: Shift the ENDO regressors that are already in X, X'X,
                     provided that there is something to shift (i.e. provided
                     that the lag length is at least 2)"""
        if self.lag_length > 1:
            self.X_t[:,self.insertion_position:] = ( 
                self.X_t[:,self.extraction_list])
            self.XX[self.insertion_position:,self.insertion_position:] = (
                self.XX[self.extraction_list,:][:,self.extraction_list])
        
        """STEP 2: For each variable x_i that newly arrives per time step, 
                   put it into X and compute relevant cross-prods X'X & X'Y"""
        i = 0
        #DEBUG: Not correct anymore for weak coupling!
        if (not (self.restriction_sequence is None) or 
            self.general_nbh_coupling == "strong coupling"):
            num_new_vars = len(self.endo_vars[0]) + self.num_exo_regressors
            new_vars = sum(self.exo_vars,[]) + self.endo_vars[0]
        elif self.general_nbh_coupling == "weak coupling":
            new_vars = (sum(self.exo_vars,[]) + 
                        sum(self.endo_vars[:self.S1*self.S2], []))
            num_new_vars = int(len(new_vars))
            
                                              
        """NEW VARIABLE LOOP I"""
        for regressor_code in new_vars: #sum(self.exo_vars,[]) + self.endo_vars[0]:
            
            """STEP 2.1: Retrieve the values of x_i"""
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo)
            else:
                x_i = self.get_endo_regressors(regressor_code, 1, 
                                            y_tm1.reshape(1, self.S1, self.S2),
                                            padding_column)
            
            """STEP 2.2: Store x_i inside X"""
            self.X_t[:,i] = x_i
            
                   
            """STEP 2.3: Compute the cross-products x_i^Tx_j for all other
                         new variables x_j s.t. j>=i as well as for all old 
                         x_j that we retain, and store in X'X_t"""
                         
            """NEW VARIABLE LOOP II"""
            for j in range(0, num_new_vars):
                if (i <= j):
                    if (j <= self.num_exo_regressors-1):
                        x_j = self.get_exo_regressors(self.all_vars[j], 
                                                      j, x_exo)
                    else:
                        x_j = self.get_endo_regressors(self.all_vars[j], 
                                        1, y_tm1.reshape(1,self.S1, self.S2),
                                        padding_column)
                    self.XX[i,j] = self.XX[j,i] = np.inner(x_i, x_j)
                    
         
                """STEP 2.4: Since for i=0, we will retrieve all new regressors
                             into x_j, use this to directly fill in the new
                             cross-products between old regressors in X and 
                             the new regressors x_j"""
                             
                """OLD VARIABLE LOOP"""
                if i == 0:
                    for k in range(num_new_vars, self.num_regressors):
                        x_k = self.X_t[:,k]
                        self.XX[k,j] = self.XX[j,k] = np.inner(x_j, x_k)           
            
            """STEP 2.5: Advance the counter"""
            i = i+1
            
        
        """STEP 3: Add X'X [X'Y, Y'Y] to X'X(r-1,t-1) [X'Y(r-1,t-1), 
                    Y'Y(r-1,t-1)]to update to X'X(r,t) [X'Y(r,t), Y'Y(r,t)]"""
        
        """STEP 3.1: Add X'X [X'Y, Y'Y] to all the t+1 possible run-lenghts"""
        self.YY = np.inner(y_t, y_t)
        self.XY = np.matmul(np.transpose(self.X_t), y_t)        
        self.XX_rt = self.XX_rt + self.XX
        self.XY_rt = self.XY_rt + self.XY
        self.YY_rt = self.YY_rt + self.YY

        """STEP 3.2: Insert X'X [X'Y, Y'Y] at position r=0 of XX_rt 
                     [XY_rt, YY_rt]. Note: For X'Y, we need to add the prior
                     influence encoded in D^-1 * beta_0. Since we always add 
                     this at r=0, we can ignore it when we add X'Y to the 
                     longer run-lengths in STEP 3.1"""
        self.XX_rt = np.insert(self.XX_rt, 0, (self.XX + self.D_inv), axis = 0)
        self.XY_rt = np.insert(self.XY_rt, 0, (self.XY + self.D_inv_b0), axis = 0)  
        self.YY_rt = np.insert(self.YY_rt, 0, self.YY, axis = 0)


    #DEBUG: Assumes acess to x_exo(t+1) at time t. General problem for BVAR
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors 
        at time point t, for all possible run-lengths."""
        
        #post_mean = np.zeros((np.size(self.retained_run_lengths), self.S1*self.S2))
        #run_length_num = self.retained_run_lengths.shape[0]
        #for r in range(0,run_length_num):
        #    post_mean[r,:] = np.matmul((self.X_tp1), self.beta_rt[r,:])

        post_mean = np.matmul((self.X_tp1), 
                              self.beta_rt[:,:,np.newaxis])
        """If you have r=t-1 twice, take post_mean twice (corr. to having a 
        CP before time t-1)"""
        #if self.retained_run_lengths[-1] == self.retained_run_lengths[-2]:
        #    post_mean = np.append(post_mean,post_mean[-1], axis=0 )
        
        return post_mean

    #DEBUG: Assumes acess to x_exo(t+1) at time t. General problem for BVAR    
    #DEBUG: Are we getting the posterior variance or its inverse here?!
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at 
        time point t, for all possible run-lengths."""

        post_var = np.zeros((np.size(self.retained_run_lengths), 
                                     self.S1*self.S2, self.S1*self.S2))
        
        """NOTE: See the derivations in my notes"""
        run_length_num = self.retained_run_lengths.shape[0]
        for r in range(0,run_length_num):
            
            """Get inverse using stored quantities & woodbury"""
            a_ = self.a + (r+1.0)*0.5
            b_ = (self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt[r] - 
                            self.beta_XX_beta_rt[r]))
            
            #NOTE: Overflow encountered here!
            post_var[r,:,:] = (b_/a_)*(np.identity(self.S1*self.S2) + 
                       np.matmul(self.X_tp1, np.matmul(self.M_inv_1_rt[r,:,:], 
                                            np.transpose(self.X_tp1))))  
        
        #if self.retained_run_lengths[-1] == self.retained_run_lengths[-2]:
        #    post_var[-1] = np.insert(post_mean, 0, post_mean[-1], axis=0 )
        
        return post_var
    
    def prior_update(self,t, model_specific_rld):
        """Update b and a priors using a_, b_ as well as beta mean prior using
        beta_rt"""
        
        #DEBUG: We lose a lot of information by exp-conversion
        filled_in = model_specific_rld > -np.inf
        """Update 1: Regression coefficients"""
        if True:
            self.prior_mean_beta = np.sum(self.beta_rt * 
                                    np.exp(model_specific_rld[filled_in,np.newaxis]), axis=0)
            self.D_inv_b0 = np.matmul(self.D_inv, self.prior_mean_beta)
            self.b0_D_inv_b0 = np.inner(self.prior_mean_beta, self.D_inv_b0)
        """Update 2: variance parameters"""
        if True:
            a_vec = self.a + (self.retained_run_lengths+1.0)*0.5
            b_vec = (self.b + 0.5*(self.b0_D_inv_b0 + 
                    self.YY_rt + self.beta_XX_beta_rt)) 
            self.a = np.inner(a_vec, np.exp(model_specific_rld[filled_in]))
            self.b = np.inner(b_vec, np.exp(model_specific_rld[filled_in]))

            
            
    def get_x_new(self,y_t, x_exo_tp1, t, padding_column=None):
        """STEP 1: Shift the ENDO regressors that are already in X, X'X,
                     provided that there is something to shift (i.e. provided
                     that the lag length is at least 2)"""
        if self.lag_length > 1:
            x_new = np.zeros((self.S1*self.S2, self.num_regressors))
            x_new[:,self.insertion_position:] = (
                    self.X_t[:,self.extraction_list].copy())
        else:
            x_new = np.zeros((self.S1*self.S2, self.num_regressors))

        """STEP 2: For each variable x_i that newly arrives per time step, 
                   put it into X and compute relevant cross-prods X'X & X'Y"""
        i = 0
                                              
        """NEW VARIABLE LOOP I: Only over exogeneous variables and first lag"""
        #DEBUG: This is a list of lists! We want a list of codes!
        for regressor_code in sum(self.exo_vars,[]) + self.endo_vars[0]:
            
            """STEP 2.1: Retrieve the values of x_i"""
            if i <= self.num_exo_regressors - 1:
                x_i = self.get_exo_regressors(regressor_code, i, x_exo_tp1)
            else:
                #DEBUG: y can NOT be a three-dim array! But its treated as one
                #       in get_endo_regr...
                #       We definitely need to re-format it s.t. it has S1xS2 as
                #       dimension
                #       Solution 1: Give y_t an additional np.newaxis
                #       Solution 2: Change stuff inside endo_regressors
                x_i = self.get_endo_regressors(regressor_code, 1, 
                                               y_t.reshape(1,self.S1, self.S2),
                                               padding_column)
            
            """STEP 2.2: Store x_i inside X"""
            x_new[:,i] = x_i
            i = i+1
            
        return x_new
    
    
#    def prior_update(self, t, r_list=None):
#        """update the prior expectation & variance to be the posterior 
#        expectation and variances weighted by the run-length distribution"""
#        self.pred_exp = np.sum((self.get_posterior_expectation(t) * 
#             (np.exp(self.joint_log_probabilities - self.model_log_evidence)
#             [:,np.newaxis, np.newaxis])), axis=0)
#        """cannot use get_posterior_variance here, because that returns the
#        covariance matrix in the global format. I need it in the local (Naive)
#        format here though"""
#        posterior_variance = self.suff_stat_var 
#        self.pred_var = np.sum(posterior_variance *
#             np.exp(self.joint_log_probabilities - self.model_log_evidence)
#             [:,np.newaxis, np.newaxis], axis=0)
#        """finally, update the prior mean and prior var"""
#        self.prior_mean = self.pred_exp
#        self.prior_var = self.pred_var
        
        
    def trimmer(self, kept_run_lengths, BAR_submodel = False):
        """Trim the relevant quantities for the BVAR NIG model"""
        
        """If this BVAR model is a submodel of a BAR model, its joint log
        probs will never be updated/grown. In that case, all it is needed for 
        is the evaluation of predictive probabilities"""
        if not BAR_submodel:
            self.joint_log_probabilities = (
                    self.joint_log_probabilities[kept_run_lengths])
        """If we optimize hyperparameters, we also want to discard the deriva-
        tives that are not warranted."""
        if self.hyperparameter_optimization:
            self.model_specific_joint_log_probabilities_derivative_sign = (
                self.model_specific_joint_log_probabilities_derivative_sign[:,
                        kept_run_lengths])
            self.model_specific_joint_log_probabilities_derivative = (
                self.model_specific_joint_log_probabilities_derivative[:,
                        kept_run_lengths])
        
        """Discard all quantities of data that have been computed"""
        self.beta_rt = self.beta_rt[kept_run_lengths,:]
        self.beta_XX_beta_rt = self.beta_XX_beta_rt[kept_run_lengths]
        self.XX_rt = self.XX_rt[kept_run_lengths,:,:]
        self.XY_rt = self.XY_rt[kept_run_lengths,:]
        self.YY_rt = self.YY_rt[kept_run_lengths]
        #self.log_det_rt = self.log_det_rt[kept_run_lengths]
        #self.R_rt = self.R_rt[kept_run_lengths,:,:]
        #self.Q_rt = self.Q_rt[kept_run_lengths,:,:]
        self.M_inv_1_rt = self.M_inv_1_rt[kept_run_lengths,:,:]
        self.M_inv_2_rt = self.M_inv_2_rt[kept_run_lengths,:,:]
        self.log_det_1_rt = self.log_det_1_rt[kept_run_lengths]
        self.log_det_2_rt = self.log_det_2_rt[kept_run_lengths]
        self.retained_run_lengths = (
                    self.retained_run_lengths[kept_run_lengths])
        self.model_log_evidence = scipy.misc.logsumexp(
                        self.joint_log_probabilities )
        
    #DEBUG: Either compute X_2 in here, or call the same fct. using different
    #       arguments
    def compute_X_XX_XY_YY(self, Y0, X0_endo, X0_exo, padding_columns = None):
        """Compute X'X from scratch. Called whenever we compute X'X(r=0, t)
        Note that the nbh_strings will be a list like this:
            
            [["topleft", "top", "topright", "left", "center", "right", 
              "bottomleft", "bottom", "bottomright"], 
             ["top", "left", "center","right"], 
             ["intercept", "center"]]
        
        """

        """Computation: Loop over both exogeneous and endogeneous variables, 
        retrieve their cross-products. If you have already computed the cross
        product before, just copy the relevant entry in X'X and paste it."""
        
        #DEBUG: Reshape X0_endo into (lag_length,S1, S2)
        X0_endo = X0_endo.reshape(self.lag_length, self.S1, self.S2)
        
        lag_count1, lag_count2 = 0,0
        
        """OUTER LOOP: Over all regressors"""
        for i in range(0, self.num_regressors):
            
            """Since exo vars are stored first in all_vars, this condition 
            allows us to see if we need to access exo or endo vars"""
            if (i <= (self.num_exo_regressors - 1)): 
                """EXOGENEOUS"""
                data_vector1 = self.get_exo_regressors(self.all_vars[i], i,
                                                       X0_exo)
            else:
                """If we need endo vars, make sure that we advance the lag
                length appropriately afterwards"""
                if (i >= self.lag_counts[lag_count1]):
                    lag_count1 = lag_count1 + 1
                """ENDOGENEOUS"""
                """I.e., if we do not pass padding columns, we cannot access
                the None-type object and thus skip the argument"""
                if padding_columns is None:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i], 
                                                     lag_count1, X0_endo)
                else:
                    data_vector1 = self.get_endo_regressors(self.all_vars[i], 
                                    lag_count1, X0_endo, padding_columns[i,:])
 
            lag_count2 = 0 #reset lag count
            
            """INNER LOOP: Over all regressors"""
            for j in range(0, self.num_regressors):
                
                if (i <= j):
                    """Compute only if you have not done so yet!"""
                    if (j <= (self.num_exo_regressors - 1)):
                        """EXOGENEOUS"""
                        data_vector2 = self.get_exo_regressors(
                                self.all_vars[j], j, X0_exo)
                    else:
                        """If we need endo vars, make sure that we advance the lag
                        length appropriately afterwards"""
                        if (j >= self.lag_counts[lag_count2]):
                            lag_count2 = lag_count2 + 1
                        """ENDOGENEOUS"""
                        if padding_columns is None:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo)
                        else:
                            data_vector2 = self.get_endo_regressors(
                                self.all_vars[j], lag_count2, X0_endo, 
                                padding_columns[i,:])
                        
                    
                    """if i == 0, we loop over all j. Use this to compute X'Y
                    as well as X"""
                    if(i == 0):
                        self.X_t[:,j] = data_vector2
                        self.XY[j] = np.inner(data_vector2, Y0)
                                 
                    """Computation: Fill in X'X with dot products!"""
                    prod = np.inner(data_vector1, data_vector2)
                    self.XX[i,j] = prod
                    self.XX[j,i] = prod
        
        """Lastly, compute Y'Y"""
        self.YY = np.inner(Y0, Y0)
    
    
    def get_exo_regressors(self, regressor_code, i, data):
        """Extract all the exogeneous regressors as you should"""
        
        """If the regressor_code is intercept, you just create intercepts.
        Otherwise, you just extract the relevant row in data"""
        #DEBUG: Check that data has format (num_exos, S1,S2)!
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2)) #assumes ONE intercept
        elif self.intercept_codes != ["intercept"]:
            """retrieve the number of the group in question"""
            group_number = int(regressor_code.split("_")[-1])
            """return the intercept only at the desired locations"""
            data_vector = self.intercept_grouping[group_number,:,:].flatten()
        else:
            data_vector = data[i,:,:].flatten()
        return data_vector.flatten()
    
    
    def get_endo_regressors(self, regressor_code, lag, data, 
                            padding_column=None):
        """Get the predictors in order, where we give back the *data* as it 
        should be. In particular, give the *lag*-th lattice slide with the 
        neighbourhood as specified by *position*.
        Around the edges, the *padding* is applied. The argument 
        *padding_column* is only for the demo object BVAR_NIG_CSurf: It is the
        rhs or lhs column of the lattice.
        
        NOTE: The lag passed to this function is the actual lag - 1.
              The data passed to this function at time t has times t-l:t-1
        """
        padding = self.padding   
        lag = -(lag - 1) #need this conversion since we have 1,2,3...,T order,
                         #but want to access the l-th lag, i.e. T-l. Also, the 
                         #FIRST lag corresponds to 0, i.e. T-1 i.e. the last 
                         #entry contained in the endogeneous regressors
        
        """STEP 1: Compute padding for rows, columns, and corners"""
        if padding == 0 or padding == "zero":
            padding_row = np.zeros(self.S2)
            padding_col = np.zeros(self.S1)
            padding_corners = 0.0
        elif padding == "overall_mean":
            mean = np.mean(data[lag,:,:])
            padding_row = mean * np.ones(self.S2) #np.mean(data[lag,:,:], axis=0)
            padding_col = mean * np.ones(self.S1)
            padding_corners = mean
        elif padding == "row_col_mean":
            padding_row = np.mean(data[lag,:,:], axis=0)
            padding_col = np.mean(data[lag,:,:], axis=1)
            weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
            padding_corners = (weight*np.sum(padding_row) + 
                               (1-weight)*np.sum(padding_col))

        elif padding.split("_")[-1] == "rhs" or  padding.split("_")[-1] == "lhs":
            """I.e., if we have a CSurf object, we need some extra care at the 
            boundaries of the change surface"""
            padding_row = np.mean(data[lag,:,:], axis=0)
            
            if padding.split("_")[-1] == "rhs":
                """get padding for cols as usual + specific one for rhs, lhs"""    
                padding_rhs = padding_column
                padding_lhs = padding_col = np.mean(data[lag,:,:], axis=1)
                weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = (weight*np.sum(padding_row) + 
                               (1-weight)*np.sum(padding_rhs))
                padding_corner_lhs = padding_corners = (
                        weight*np.sum(padding_row) + 
                               (1-weight)*np.sum(padding_lhs))
            else:
                """get padding for cols as usual + specific one for rhs, lhs"""
                padding_rhs = padding_col = np.mean(data[lag,:,:], axis=1)
                padding_lhs = padding_column
                weight = (np.size(padding_row)/
                      (np.size(padding_row) + np.size(padding_col)))
                padding_corner_rhs = padding_corners = (weight*
                        np.sum(padding_row) + (1-weight)*np.sum(padding_rhs))
                padding_corner_lhs = (weight*np.sum(padding_row) + 
                               (1-weight)*np.sum(padding_lhs))
            
        #data_vector = np.ones((self.S1, self.S2))
        """STEP 2A: Get the data_vector for the 4-nbh case or intercept/center"""
        #DEBUG: intercept will be called in exo regr, this is redundant
        if regressor_code == "intercept":
            data_vector = np.ones((self.S1, self.S2))
        elif regressor_code == "center":
            data_vector = data[lag,:,:]
        elif regressor_code == "left": 
            if padding.split("_")[-1] == "rhs": 
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag,:,:-1], 0, padding_col, axis=1) 
                
        elif regressor_code == "right":
            if padding.split("_")[-1] == "lhs":
                """Insert the padding column passed to this function"""
                data_vector = np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1)
            else:
                """Take the row averages as padding"""
                data_vector = np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1)
                
        elif regressor_code == "top":
            data_vector = np.insert(data[lag,:-1,:], 0, padding_row, axis=0)
        elif regressor_code == "bottom":
            data_vector = np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0)
            
        elif regressor_code == "topleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, 1:] = data[lag,:-1,:-1]
            if padding.split("_")[-1] == "rhs": 
                """Insert the padding column passed to this function"""
                data_vector[0,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corner_rhs, padding_rhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])
 

        elif regressor_code == "topright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[1:, :-1] = data[lag,:-1,1:]
            if padding.split("_")[-1] == "lhs": 
                """Insert the padding column passed to this function"""
                data_vector[0,:] = np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] = np.append(padding_corner_lhs, padding_lhs[:-1])
            else:
                """Take the row averages as padding"""
                data_vector[0,:] = np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] = np.append(padding_corners, padding_col[:-1])
            
        elif regressor_code == "bottomleft":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, 1:] = data[lag,1:,:-1]
            if padding.split("_")[-1] == "rhs": 
                """Insert the padding column passed to this function"""
                data_vector[-1,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append( padding_rhs[1:],padding_corner_rhs)
            else:
                """Take the row averages as padding"""
                data_vector[-1,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_col[1:], padding_corners)
            
            
        elif regressor_code == "bottomright":
            data_vector = np.zeros((self.S1, self.S2))
            data_vector[:-1, :-1] = data[lag,1:,1:]
            if padding.split("_")[-1] == "lhs": 
                """Insert the padding column passed to this function"""
                data_vector[-1,:] = np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] = np.append(padding_lhs[1:],padding_corner_lhs )
            else:
                """Take the row averages as padding"""
                data_vector[-1,:] = np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] = np.append(padding_col[1:], padding_corners)
            
        elif regressor_code == "4_inner_nbh_res":
            if padding.split("_")[-1] == "lhs": 
                #pad with real data on the right
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            else:
                data_vector = (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))

        elif ((regressor_code == "4_outer_nbh_res") or 
              (regressor_code == "8_nbh_res")):
            if padding.split("_")[-1] == "lhs": 
                #pad with real data on the right
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])  
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] += np.append(padding_corner_lhs, padding_lhs[:-1]) 
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append( padding_row[1:], padding_corner_lhs)
                data_vector[:,-1] += np.append(padding_lhs[1:],padding_corner_lhs )
            elif padding.split("_")[-1] == "rhs":
                #pad with real data on the left
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corner_rhs, padding_rhs[:-1])
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_corners, padding_col[:-1]) 
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corner_rhs, padding_row[:-1])
                data_vector[:,0] += np.append( padding_rhs[1:],padding_corner_rhs)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_col[1:], padding_corners)
            else:
                """initialize with topleft"""
                data_vector = np.zeros((self.S1, self.S2))
                data_vector[1:, 1:] = data[lag,:-1,:-1]
                data_vector[0,:] = np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] = np.append(padding_corners, padding_col[:-1])  
                """add topright"""
                data_vector[1:, :-1] += data[lag,:-1,1:]
                data_vector[0,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_corners, padding_col[:-1]) 
                """add bottomleft"""
                data_vector[:-1, 1:] += data[lag,1:,:-1]
                data_vector[-1,:] += np.append(padding_corners, padding_row[:-1])
                data_vector[:,0] += np.append(padding_col[1:], padding_corners)
                """add bottomright"""
                data_vector[:-1, :-1] += data[lag,1:,1:]
                data_vector[-1,:] += np.append(padding_row[1:], padding_corners)
                data_vector[:,-1] += np.append(padding_col[1:], padding_corners) 

        if regressor_code == "8_nbh_res":
            """Will already have the outer 4-nbh stored inside of data_vector
            due to the previous elif-bracket call!"""
            if padding.split("_")[-1] == "lhs": 
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_lhs, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            elif padding.split("_")[-1] == "rhs":
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_rhs, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
            else:
                data_vector += (np.insert(data[lag,:,:-1], 0, padding_col, axis=1) +
                    np.insert(data[lag,:,1:], self.S2-1, padding_col, axis=1) + 
                    np.insert(data[lag,:-1,:], 0, padding_row, axis=0) + 
                    np.insert(data[lag,1:,:], self.S1-1, padding_row, axis=0))
        
        """STEP 2B: Get the data_vector for the general nbh case"""
        """Note: No need to extract the lag, since that is already passed
            to the function as *lag*. But we do need to know which group 
            we need to extract the regressors from. We want the g-th group
            for each neighbourhood, so we access the g-th element in 
            [[[1,2,3],[4,5],[6]], [[3,4],[5,6,7],[8,9]], ... ], i.e. if g=0
            we have relevant_nbh = [1,2,3], then relevant_nbh = [3,4], etc."""
        if regressor_code.split("_")[0] == "general":
            data_vector = np.zeros((self.S1, self.S2))
            
            """STEP 2B.a: If we have strong coupling, all nbhs N_i(s) for a
            specific i but pooled across space have the same effect beta_i for 
            each location inside N_i(s)."""
            if self.general_nbh_coupling == "strong coupling":
                """In this case, we have the same input as for weak
                coupling, but a different interpretation. In particular, we couple
                the effects over different spatial locations. Accordingly, we save
            
                general_nbh_<lag>_<nbh_index> only.
            
                Then, in the extractors, we loop over <loc> to retrieve the 
                regressors in a single column as
            
                regressor(<lag>, <nbh_index>)[<loc>] = sum over all measurements
                    at time t - <lag> for nbh given by
                    gen_nbh_seq[<loc>][gen_nbh_res_seq[<lag>][<nbh]].
                """
            #if self.general_nbh_coupling == "strong coupling":
                data_vector = data_vector.flatten()
                """num_group retrieves the index i of N_i(s)"""
                num_group = (regressor_code.split("_")[-1])
                
                if num_group == "center":
                    #We just need the old obs of same location
                    data_vector = data[lag,:,:].flatten()
                else:
                    """convert from str to int"""
                    num_group = int(num_group)
                    """relevant_nbhs retrieves the collection 
                    {N_i(s): s \in S1*S2}"""
                    relevant_nbhs = [item[num_group] for item in 
                             self.general_nbh_sequence]
                    """Since we assume the s' \in {N_i(s): s \in S1*S2} to have the
                    same linear effect, sum them all up s.t. we put 
                    sum(s' \in N_i(s)) in position i"""

                    dat = data[lag,:,:].flatten()
                    print(relevant_nbhs)
                    print([sum(dat[relevant_nbh]) for relevant_nbh
                                    in relevant_nbhs])
                    print(np.array([sum(dat[relevant_nbh]) for relevant_nbh
                                    in relevant_nbhs]))
                    print(data.vector.shape)
                    print(np.array([sum(dat[relevant_nbh]) for relevant_nbh
                                    in relevant_nbhs]).shape)
                    data_vector = np.array([sum(dat[relevant_nbh]) for relevant_nbh
                                    in relevant_nbhs])

            elif self.general_nbh_coupling == "weak coupling":
                """STEP 2B.b: If we have weak coupling, all s' \in N_i(s) for a
                specific i and a specific space have the same effect beta_is for 
                each location inside N_i(s)."""
                data_vector = data_vector.flatten()
                """num_group retrieves the index i of N_i(s) OR 'center' for 
                the autoregressive term"""
                if regressor_code.split("_")[-2] == "center":
                    #retrieve AR term
                    location = int(regressor_code.split("_")[-1])
                    data_vector[location]= (data[lag,:,:].flatten()
                            [location])
                else:
                    """What we want here is retrieve the data corr. to the 
                    <nbh>-th nbh of location <loc> at lag <lag> +1. This
                    is done by getting
                    general_nbh_sequence[<loc>][general_nbh_restriction_seq
                                        [<lag>][<nbh_index>]]
                    from strings of form
                    'general_nbh_sequence_<lag>_<nbh_index>_<loc>'
                    """
                    nbh_index = int(regressor_code.split("_")[-2])
                    #DEBUG: will be the same as lag (we hope)
                    lag2 = int(regressor_code.split("_")[-3])
                    #print(lag)
                    #print(nbh_index)
                    relevant_nbh_index = self.general_nbh_restriction_sequence[lag2][nbh_index]
                    """location retrieves the s of N_i(s)"""
                    location = int(regressor_code.split("_")[-1])
                    """relevant_nbh retrieves the indices s' \in N_i(s)"""
                    #print(self.general_nbh_sequence[location][relevant_nbh_index])
                    relevant_nbh = (self.general_nbh_sequence[location][relevant_nbh_index])
                    """now, the entire data vector is 0 except for the index corr.
                    to location s, where we store sum(s' \in N_i(s))"""
                    data_vector = np.zeros(self.S1*self.S2)
                    """if relevent_nbh = [], this will be FALSE and we return
                    a vector of zeros in that case."""
                    if relevant_nbh:
                        data_vector[location]= sum(data[lag,:,:].flatten()
                            [relevant_nbh])
                
        """STEP 3: Return the data vector"""
        return data_vector.flatten()
    
    
    #DEBUG: Put into probability_model, since used by BVAR and NIW.
    @staticmethod
    def mvt_log_density(y_flat, prec, log_det, df, prior = False, alerts = False):
        """Returns the density of a multivariate t-distributed RV.
        Assumes that the mean is 0.
        
        Here, we have y_flat being the point at which we want to evaluate the
        density, mu its mean, prec the precision matrix, and det the cov matrix'
        determinant. A very helpful reference is the formulation in
        https://www.statlect.com/probability-distributions/
        multivariate-student-t-distribution
        """      
        p, nu = y_flat.shape[0], df
        #DEBUG: 1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat)) is 
        #       negative sometimes, and it should NOT be. (because prec
        #       should be positive definite!)
        log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
        if( log_term<0 or np.isnan(log_det) ):
            #print("multiplication part: ",np.matmul(np.matmul(y_flat, prec),y_flat))
            #print("r = :" len(y_flat)/prec.shape[0])
            #log_term = np.inf
            if not prior:
                if alerts:
                    print("covariance estimate not s.p.d. or log_det nan")
                    print("degrees of freedom: ", df)
                #DEBUG: only do nearest pd matrix if not prior, otherwise
                #       just add an identity matrix that is large enough
                prec = (NPD.nearestPD(prec) + np.identity(prec.shape[0])*max(df*nu, 25))
                log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
            elif prior:
                #DEBUG: First try the easy fix, then the computationally more expensive one
                if log_term<0:
                    prec = prec + np.identity(prec.shape[0])*nu*df
                    log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
                if log_term<0:
                    prec = (NPD.nearestPD(prec) + np.identity(prec.shape[0])*max(df*nu, 25))
                    log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
                count = 0
                while log_term<0:
                    if count == 0:
                        print("Covariance matrix injected with sphericity")
                    prec = prec + np.identity(prec.shape[0])*nu*df
                    log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
                    count = count+1
            #prec = (NPD.nearestPD(prec)+nu*df*np.identity(prec.shape[0]))
            #log_term = (1+(1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat))
            if( log_term<0):
                print("non-s.p.d. covariance estimate:",
                      "problem persists! Set it to log(pow(10,-100))")
                log_term = np.log(pow(10,-50))
                #_, log_det = np.linalg.slogdet(prec)
                log_det = np.log(pow(100, p)) # = slogdet(np.identity*100)
            else:
                log_term = np.log(log_term)
                _, log_det = np.linalg.slogdet(prec)
            if np.isnan(log_det):
                print("log_det nan: problem persists!")
            
        else:
            log_term = np.log(log_term)
            
        """Note: Should not happen after we have corrected b_ to be positive"""
        if np.isnan(log_det):
            print("nan log det")
            _, log_det = np.linalg.slogdet(prec)
            log_det = 1.0/log_det #since we want the log det of cov mat
            if np.isnan(log_det):
                print("problem persists!")
            
        
        calc = (special.gammaln(0.5*(nu+p)) - special.gammaln(0.5*nu) -
                 0.5*p*( np.log(nu) + np.log(np.pi) ) - 0.5*log_det -
                 0.5*(nu+p)*log_term)
        #is calc nan?
        if np.isnan(calc):
            print("Alert! Calc is nan")
            calc = -pow(10,-15)
                 #np.log(1+ 
                 #    (1.0/nu)*np.matmul(np.matmul(y_flat, prec),y_flat)))
        return calc
    
    
    def differentiate_predictive_log_distribution(self, y,t, 
                                                  run_length_log_distro): 
        """Take derivative of log-posterior predictive w.r.t. hyperpars a,b.
        The size of the output will be num_params x run_length_num"""
        
        #DEBUG: We need to add the derivative for r=0, too!
        
        """STEP 1: Compute expressions needed for further computation"""
        y = y.flatten()
        p = np.size(y)
        run_length_num = self.retained_run_lengths.shape[0]
        num_params = 2 #a,b
        
        a_vec = np.insert(self.a + (self.retained_run_lengths+1.0)*0.5, 
                          0, self.a)
        b_vec = np.insert(
            self.b + 0.5*(self.b0_D_inv_b0 + self.YY_rt - self.beta_XX_beta_rt),
            0, self.b) 
        resids = np.insert(np.matmul(self.X_t,self.beta_rt[:,:,np.newaxis]) - 
                y.flatten()[:,np.newaxis],0, 
                np.matmul(self.X_t, self.prior_mean_beta[:,np.newaxis]) - 
                y.flatten()[:,np.newaxis], axis=0)
#        print("resids", resids.shape)
#        print("self.C_t_inv", self.C_t_inv.shape)
        products = np.sum(resids * np.matmul(self.C_t_inv, resids), axis=1).reshape(run_length_num+1)
            
        
        """STEP 2: Compute a-derivative in log form"""
        expr1 = scipy.special.psi(a_vec + 0.5*p)
        expr2 = -scipy.special.psi(a_vec)
        expr3 = -p/(2.0*a_vec) #expr4 = 0
        expr5 = 0.5*self.S1*self.S2*(1.0/a_vec)
        
        """Note: The inside of expression 6 could be negative, so we store its
        sign and take the log of its absolute value. We then pass the signs on
        so that the actual value of the gradient can be constructed as 
        sign(inside6)*np.exp(gradient)"""
#        print("resids", resids.shape)
#        print("expr1", expr1.shape)
#        print("expr2", expr2.shape)
#        print("expr3", expr3.shape)
#        print("expr5", expr5.shape)
#        #print("expr6", expr6.shape)
        inside6 = (1.0 + 0.5 * (1/b_vec) * products)
        expr6 = np.log(np.abs(inside6))
        log_posterior_predictive_gradients_a_val = (expr1 + expr2 + expr3 +
                                                expr5 + expr6)
         
        """STEP 3: Comute b-derivative in log form"""
        expr5_ = -0.5*self.S1*self.S2*(1.0/b_vec)
        expr6_ = (0.5*(p + 2*a_vec))*(1/b_vec)*(
                    0.5*products/(1+0.5*(1/b_vec)*products))
        log_posterior_predictive_gradients_b_val = expr5_ + expr6_
        
        """STEP 4: Convert d/dtheta log(f(a,b)) = f'(a,b)/f(a,b) into what 
        you actually want: log(f'(a,b)) = log( log(f(a,b)) * f(a,b) ) = 
        log(f'/f) + log(f) = log('log_posterior_predictive_gradients_val) + 
        self.evaluated_log_predictive_distr. We need to take special care 
        about the signs for the a-gradient, since we already have the sign
        of inside6, and we need to multiply it with the sign of the log-form
        log_posterior_predictive_gradients_a_val."""
#        print("products", products.shape)
#        print("expr6", expr6.shape)
#        print("expr6_", expr6_.shape)
#        print("log_posterior_predictive_gradients_a_val", log_posterior_predictive_gradients_a_val.shape)
#        print("log_posterior_predictive_gradients_b_val", log_posterior_predictive_gradients_b_val.shape)
        
        """STEP 4.1: log gradients' signs"""
        log_gradients_a_sign = (np.sign(inside6)* 
            np.sign(log_posterior_predictive_gradients_a_val))
        log_gradients_b_sign = np.sign(
            log_posterior_predictive_gradients_b_val)
        log_gradients_sign = np.array([log_gradients_a_sign, 
                            log_gradients_b_sign]).reshape(num_params, run_length_num+1)
                        
        """STEP 4.2: log gradients' values"""
        all_predictives = np.insert(
            self.one_step_ahead_predictive_log_probs, 0, self.r0_log_prob)
        log_gradients_a_val = (np.log(
                np.max(
                    np.array([
                        np.abs(log_posterior_predictive_gradients_a_val),
                               0.000005 * np.ones((run_length_num+1))]
                    ), axis=0)
                ) + all_predictives)
        log_gradients_b_val = (np.log(
                np.max(
                    np.array([
                        np.abs(log_posterior_predictive_gradients_b_val),
                               0.000005 * np.ones((run_length_num+1))]
                    ), axis=0)
                ) + all_predictives)
#        log_gradients_a_val = (np.log(
#            np.abs(log_posterior_predictive_gradients_a_val)) +
#            all_predictives)
#        log_gradients_b_val = (np.log(
#            np.abs(log_posterior_predictive_gradients_b_val)) +
#            all_predictives)
        log_gradients_val = np.array([log_gradients_a_val, 
                              log_gradients_b_val]).reshape(num_params, run_length_num+1)       
        
        """STEP 5: Finally, return the log gradients and the signs"""
        return log_gradients_val, log_gradients_sign
    
    
    def caron_hyperparameter_optimization(self, t, gradient, step_size):
        """Gradient for type-II ML for the hyperparameters a and b, as in 
        Caron, Doucet, Gottardo (2012).
        Called by ProbabilityModel level at each time step"""  
        
        #if t-self.lag_length > 30:
        """STEP 2: Get the difference a_{t-1} - a_{t-2}, b_{t-1} - b_{t-2}"""
        #Debug: Initialize self.a_old as self.a!, gradient_old = 0
#        if self.a == self.a_old and self.b == self.b_old:
#            disturbance = np.random.normal(loc=0.0, scale=pow(10,-6))
#        else:
        scale = -10
        max_step_scale = pow(10,-3)*5#0.05
        min_step_scale = pow(10,-4)#0.001
        disturbance = 0.0 #np.random.normal(loc=0.0, scale=pow(10,scale)) #
        dif_old_a = self.a - self.a_old  + disturbance
        dif_old_b = self.b - self.b_old  + disturbance
        dif_old_val = np.array([dif_old_a, dif_old_b]) 
        dif_old_grad = gradient - self.gradient_old 
        
        """STEP 3: Compute step size via BB method"""
        if True:
            """These conversions are necessary for numerical stability"""
            dampener = pow((1/1.005), t*0.1)#pow(0.5, t*0.25)
            D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
            if D1_sign >0:
                D1 = min(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10,5))
            else:
                D1 = max(- np.abs(np.dot(dif_old_val, dif_old_grad)), 
                         -pow(10, 5))
            if np.abs(D1) < pow(10,-1):
                D1 = (pow(10,-1))*D1_sign
            D2 = min(np.dot(dif_old_grad, dif_old_grad), pow(10,5))
            if D2 < pow(10,-1):
                D2 = pow(10,-1) 
            D3 = min(np.dot(dif_old_val, dif_old_val), pow(10,5))
            if D3 < pow(10,-1):
                D3 = pow(10,-1)
            alpha_1 = ((D1/D2)*dampener)/(pow(10,-scale))
            #alpha_2 = D3/D1
#            alpha_1 = (np.dot(dif_old_val, dif_old_grad)/
#                       np.dot(dif_old_grad, dif_old_grad))
#            alpha_2 = (np.dot(dif_old_val, dif_old_val)/
#                       np.dot(dif_old_val, dif_old_grad))
            step_size_abs = max(np.abs(alpha_1),step_size) #* dampener #*dampener #*dampener #* step_size
            #allow maximum change to be half of current magnitude
            step_size = np.sign(alpha_1)*step_size_abs
            if np.sign(alpha_1) == 0.0:
                sign_a = np.sign(gradient[0])
                sign_b = np.sign(gradient[1])
            else:
                sign_a = np.sign(step_size*gradient[0])
                sign_b = np.sign(step_size*gradient[1])
                
            increment_a = (sign_a*
                max(self.a*dampener*min_step_scale, 
                min( self.a*dampener*max_step_scale, 
                    np.abs(step_size_abs*gradient[0]))))
            increment_b = (sign_b*
                max(self.a*dampener*min_step_scale, 
                min( self.b*dampener*max_step_scale, 
                    np.abs(step_size_abs*gradient[1]))))
        """STEP 2: Calculate the new a-value, and return the difference to
        old a-value, (a_{it+1} - a_it)"""
        #add disturbance term to ensure that step size changes at boundaries, too
        self.a_old = self.a #+ np.random.normal(loc=0.0, scale = 0.001)
        self.b_old = self.b #+ np.random.normal(loc=0.0, scale = 0.001)
        self.gradient_old = gradient #+ np.random.normal(loc=0.0, scale = 0.001)
        self.a = min(max(self.a + increment_a, 
                         1.0), pow(10,pow(10,3)))
        self.b = min(max(self.b + increment_b, 
                         pow(10,-20)), pow(10,pow(10,3)))
        self.a_list.append(self.a)
        self.b_list.append(self.b)
        
        
    def turner_hyperparameter_optimization(self, step_size):
        """Called by HyperparameterOpimizer at each iteration of the 
        optimization routine"""
        
        """STEP 1: Get the new gradient value"""
        sign, gradient = scipy.misc.logsumexp(
                a=(self.model_specific_joint_log_probabilities_derivative),
                b=self.model_specific_joint_log_probabilities_derivative_sign,
                return_sign=True, axis=1)
        gradient = np.exp(gradient)*sign
        
        """STEP 2: Get the difference a_{t-1} - a_{t-2}, b_{t-1} - b_{t-2}"""
        #Debug: Initialize self.a_old as self.a!, gradient_old = 0
        dif_old_a = self.a - self.a_old
        dif_old_b = self.b - self.b_old
        dif_old_val = np.array([dif_old_a, dif_old_b])
        dif_old_grad = gradient - self.gradient_old
        
        """STEP 3: Compute step size via BB method"""
        if True:
            D1_sign = np.dot(np.sign(dif_old_val), np.sign(dif_old_grad))
            D1 = min(max(np.abs(np.dot(dif_old_val, dif_old_grad)), pow(10,5)),
                     pow(10, -5))*D1_sign
            D2 = min(max(np.dot(dif_old_grad, dif_old_grad), pow(10,5)),
                     pow(10,-5))
            D3 = min(max(np.dot(dif_old_val, dif_old_val), pow(10,5)),
                     pow(10,-5))
            alpha_1 = D1/D2
            alpha_2 = D3/D1
#            alpha_1 = (np.dot(dif_old_val, dif_old_grad)/
#                       np.dot(dif_old_grad, dif_old_grad))
#            alpha_2 = (np.dot(dif_old_val, dif_old_val)/
#                       np.dot(dif_old_val, dif_old_grad))
            step_size = alpha_1 #*dampener #* step_size
        
        """STEP 2: Calculate the new a-value, and return the difference to
        old a-value, (a_{it+1} - a_it)"""
        self.a_old, self.b_old = self.a, self.b
        self.gradient_old = gradient
        self.a = min(max(self.a + gradient[0]*step_size, pow(10,-20)), pow(10,15))
        self.b = min(max(self.b + gradient[1]*step_size, pow(10,-20)), pow(10,15))
        a_dif, b_dif = self.a - self.a_old, self.b - self.b_old
        return [a_dif, b_dif]
        
        
        
    
    @staticmethod
    def objective_optimization(x, *args):
        """Objective function for type-II ML for the hyperparameters a and b.
        Can be used inside conjugate gradient descent methods like this one:
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
            scipy.optimize.fmin_cg.html
        in the way outlined in this paper: 
            http://mlg.eng.cam.ac.uk/pub/pdf/TurSaaRas09.pdf
            
        args = S1, S2, y_t, X_t (would be X_tp1 if this were called before 
                updating the predictive distros), retained_run_lengths, 
                run_length_distro, beta_rt, log_det_1, log_det_2, C_t_inv, 
                b0_D_inv_b0, YY_rt, beta_XX_beta_rt
        x = a,b
        """        
        
        """STEP 1: get *args, which are basically a heap of quantities stored
        inside BVARNIG that you have to pass to the static method"""
        (S1, S2, y, X, retained_run_lengths, run_length_distro, beta_rt, 
             log_det_1_rt, log_det_2_rt, C_t_inv, b0_D_inv_b0, YY_rt, 
             beta_XX_beta_rt) = args
        a,b = x
        y = y.flatten()
        
        """STEP 2: From the inputs, compute posterior predictive for all run
        lengths and multiply by the run-length distro specific to this 
        BVAR model (do NOT use the model-and-run-length distro). Note that
        we will have to use logsumexp, as all the MVSt-densities are in log
        format"""
        
        """STEP 2.1: Get a_ and b_ for each run-length"""
        a_vec = a + (retained_run_lengths+1.0)*0.5
        b_vec = (b + 0.5*(b0_D_inv_b0 + YY_rt - beta_XX_beta_rt)) 
        
        """STEP 2.2: Get the log determinants as function of a_ and b_"""
        log_dets = ((S1 * S2)*(np.log(b_vec) - np.log(a_vec)) + 
                           log_det_1_rt - log_det_2_rt)
        
        """STEP 2.3: Get the log densities corresponding to those values for
        all run-lengths"""
        run_length_num = retained_run_lengths.shape[0] 
        MVSt_log_densities = np.array([BVARNIG.mvt_log_density(
            y_flat = (np.matmul(X, beta_rt[r,:]) - y.flatten()), 
            prec = (a_vec[r]/b_vec[r])*C_t_inv[r,:,:], 
            log_det = log_dets[r], prior = False, alerts = False) 
            for r in range(0, run_length_num)])

        """STEP 2.4: Multiply the MVSt densities with the run_length distro,
        which is equivalent to adding them on a log scale. Afterwards, use 
        logsumexp to get the sum over all of them (again on a log scale)"""
        evaluation_objective = scipy.misc.logsumexp(
                MVSt_log_densities + run_length_distro)
        
        """STEP 3: Return the objective value"""
        return evaluation_objective
    


#if __name__ == "__main__":
#    from BVAR_NIG_Sim import BVARNIGSim
#    #from detector import Detector
#    import matplotlib.pyplot as plt
#    S1, S2, T = 2,2,100
#    mySim = BVARNIGSim(S1 = S1, S2 = S2, T = T, CPs = 1, CP_locations = [50], 
#                 sigma2 = np.array([50,4]), 
#                 mu = np.array([5, 0]), 
#                 nbh_sequences = [ [0,0], [0,0] ],
#                 restriction_sequences = [ [0,0], [0,0]],
#                 intercept=False,
#                 coefs = np.array([ 0.8*np.ones(1+1), 0.2*np.ones(1+1)]),
#                 burn_in = 100,
#                 padding = "row_col_mean")
#    mySim.generate_data()
#    plt.plot(np.linspace(1,mySim.T, mySim.T), mySim.data[:,0,0])
#    plt.show()
#    
#    data = mySim.data
    
#    myBVAR = BVARNIG(prior_a=1, prior_b=2, 
#                     prior_mean_beta=np.zeros(S1*S2), 
#                     prior_var_beta=np.identity(S1*S2),
#                     S1 = S1, S2 = S2, 
#                     separate_intercepts=False, 
#                     nbh_sequence=np.array([4]), 
#                     nbh_sequence_exo=np.array([0]), 
#                     exo_selection = [],
#                     padding = 'overall_mean', 
#                     auto_prior_update=False,
#                     restriction_sequence = np.array([4])
#                     )
    
#    myDetector = Detector(data, model_universe, model_prior, cp_model, 
#                 S1, S2, T, exo_data=None, num_exo_vars=None, threshold=None)
    