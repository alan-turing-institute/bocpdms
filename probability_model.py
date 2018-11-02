# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements the abstract class ProbabilityModel, which is the 
parent class of all probability models used by Spatial BOCD. E.g., the 
NaiveModel class has this abstract class as its parent and implements the
model where we treat each spatial location as independent from all others
(in a Gaussian setting)
"""

"""import all you need to build an abstract class"""
#from abc import ABCMeta, abstractmethod
import numpy as np
import scipy #.special #import logsumexp

class ProbabilityModel:
    #__metaclass__ = ABCMeta
    """An abstract class for all probability models that will live in the model
    universe. I.e., each explicit probability model (e.g., the naive model that
    does assume independence between locations) will inherit from this class.
    
    
    NOTE: I Should still look into superclass constructors!
    
    Abstract Methods:
        predictive_probabilities
        growth_probabilities
        evidence
        cp_probabilities
        run_length_distribution
        update_predictive_distributions
    """
    

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS     
    def evaluate_predictive_log_distribution(self, y,r):
        """Evaluate the predictive probability associated with run length
        *r*. Gives back ONE quantity reflecting the overall probability 
        density of the current observation *y* across all spatial 
        locations, given run length *r*"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t==1!     
    def initialization(self, y, cp_model, model_prior):
        """Initialize the joint_probabilities before the first iteration!"""
        pass
    
    #SUPER CLASS IMPLEMENTATION
    #CALLED BY DETECTOR
    def update_joint_log_probabilities(self, y,t, cp_model, model_prior,
                log_model_posteriors, log_CP_evidence):
        #DEBUG: model_prior should not be needed anymore. Just put it into
        #       model_posterior, position 0.
        """Is called to update growth- and CP probabilities. That means it is 
        called to obtain the joint probability for (y_{1:t}, r_t) for all 
        possible run lengths r_t = 0,1,...,t-1. 
        If t=1, a special case takes place that requires calling the 
        function 'initialization' that gives us the first joint probs.    
        The *model_prior* is the prior probability of the model. Unlike
        the np array in the Detector object, the passed argument here is
        only a scalar (!) corresponding to the correct entry of the np array
        in the detector for this particular model.
        
        NOTE: *joint_probabilities* will have size t at the
        beginning and size t+1 at the end of this function call.
        
        NOTE: Desirable to also implement SOR/SCR of Fearnhead & Liu (2007) 
        in order for this to scale in constant time!
        """


        """STEP 1: evaluate predictive probabilities for all r.
        This means that for a given a model and the current time point,
        we return the evaluations of the pdf at time t conditional on all run 
        lengths r=0,1,...t-1,>t-1 for y in log-form.
        """ 
        self.r0_log_prob = self.evaluate_log_prior_predictive(y,t)
        #predictive_log_probs = self.evaluate_predictive_log_distribution(y,t)
        predictive_log_probs = self.evaluate_predictive_log_distribution(y,t)
        #DEBUG: Here, we store the predictive distros y_t|y_1:t-1,r_t-1,m_t-1
        #NOTE: we do not want the cp prob in here because that is a NEW 
        self.one_step_ahead_predictive_log_probs = predictive_log_probs.copy()
            #np.insert(predictive_log_probs.copy(), 0, r0_log_prob)
        
        """STEP 2: update all no-CP joint probabilities, so-called 'growth 
        probabilities' to follow MacKay's naming convention. There will be t
        of them, namely for r=1,2,...,t-1, > t-1, with the last probability
        always being 0 unless we have boundary conditions allowing for
        non-zero probability that we are already in the middle of a segment
        at the first observation."""
            
            
        
        #DEBUG: evaluated log prob for r=0 is not in here! (Is only needed for
        #       the CP probability though)
        #helper_log_probabilities = (self.joint_log_probabilities + 
        #                           predictive_log_probs) 
        
        #print("predictive_log_probs ", predictive_log_probs.shape)
        #print("self.joint_log_probabilities ", self.joint_log_probabilities.shape)
        #print("log_model_posteriors ", log_model_posteriors.shape)
        #print("predictive_log_probs ", predictive_log_probs.shape)
        #print("r0 log prob", r0_log_prob)
        #print("log CP evidence", log_CP_evidence)
        growth_log_probabilities = (predictive_log_probs + 
                                    self.joint_log_probabilities + 
                                    log_model_posteriors + 
                                    np.log(1-cp_model.hazard_vector(1, t)))
        #Here, the hazard rate/r_t probability is contained in log_CP_evidence
        CP_log_prob = self.r0_log_prob + np.log(model_prior) + log_CP_evidence
        #print("r0_log_prob:", r0_log_prob)
        #print("CP log prob:", CP_log_prob)
        #print("groth probs:", growth_log_probabilities)

        """STEP 3: Get CP & growth probabilties at time t by summing
        at each spatial location over all time points and multiplying by
        the hazard rate"""
        #DEBUG: We need to use chopping of run-lengths here, too
        #DEBUG: we need the prior predictive here!
        #CP_log_prob = scipy.misc.logsumexp(helper_log_probabilities + 
        #                                  np.log(cp_model.hazard_vector(1, t)))
        #self.joint_log_probabilities = (helper_log_probabilities +
        #    np.log(1-cp_model.hazard_vector(1, t)))
        
        """Put together steps 2-3"""
        joint_log_probabilities_tm1 = self.joint_log_probabilities
        self.joint_log_probabilities = np.insert(growth_log_probabilities, 
                                                0, CP_log_prob)
        #print("self.joint_log_probabilities after update ", self.joint_log_probabilities.shape)
        
        """STEP 4: Lastly, we always want to get the new evidence for this
        particular model in the model universe, which is the sum of CP and 
        growth probabilities"""
        model_log_evidence_tm1 = self.model_log_evidence
        self.model_log_evidence = scipy.misc.logsumexp(
                self.joint_log_probabilities )
        
        
        """STEP 5: If we want to optimize the hyperparameters, do so after
        all joint log probabilities, the evidence, and so on are updated"""
        if (self.hyperparameter_optimization is not None and 
            self.hyperparameter_optimization is not False):
#            """STEP 5.0: Get the joint log probabilities P(y_1:t, r_t), 
#            ignorning in particular the influence of the model m_t. This is 
#            warranted for optimizing the hyper-parameters s.t. the data fits
#            each individual model best"""
#            self.model_specific_joint_log_probabilities = 
#            CP_log_prob_2 = 
    
            """STEP 5.2: We have P(y_1:t, r_t, m_t = m), so if we sum over all
            tuples (m_t = m, r_t = r) for fixed m and variable r, what we get
            is P(y_1:t, m_t=m), meaning P(y_1:t,r_t,m_t=m)/P(y_1:t, m_t=m) = 
            P(r_t|y_1:t, m_t)"""
            run_length_log_distro = (
                    joint_log_probabilities_tm1 - model_log_evidence_tm1)
            
            """STEP 5.1: This is different for each model, so we evaluate the 
            derivative inside the individual model objects. Note that for the
            first observation (t = lag_length+1), the gradient of the pred
            log distr will just be that of the prior."""
            #DEBUG: PROBLEM: IF THESE ARE ON LOG SCALE, THAT'S FINE, BUT THEY 
            #       MUST NOT BE THE DERIVATIVE OF THE LOG OF THE PREDICTIVE!
            #       i.e., we must not have d[log(f(x))] = f'(x)/f(x).
            #       However, since log-expressions are more stable, just
            #       use the log(f(x)) and then (1) log the result, (2)
            #       use logsumexp to add f(x) to it (on log scale) to end up
            #       with f'(x).
            #DEBUG: Make sure this returns log(f'(x)) and its signs
            gradients_log_predictive_val, gradients_log_predictive_sign = (
                    self.differentiate_predictive_log_distribution(y,t, 
                    run_length_log_distro))
            num_params = gradients_log_predictive_val.shape[0]
            run_length_num = self.retained_run_lengths.shape[0]
            
            
            """STEP 5.3: Update the derivatives of the probabilities 
            P(y_1:t, r_t|m_t=m) = P(y_1:t, r_t, m_t=m)/P(m_t=m). Notice that 
            P(m_t = m) = prior(m)."""
            #DEBUG: Check if this is correct formula if something fails
            model_specific_joint_log_probs = (
                    joint_log_probabilities_tm1 - np.log(model_prior))
            
            """STEP 5.4: Caron's gradient: vectorized update of 
            log(dP(y_t|y_1:t-1)) for all parameters over which we optimize."""
            #DEBUG: Need initialization of model_specific log probabilities derivative
            #       in some way.
            #DEBUG: Unclear if nun_length_num is too large/too small for size
            #       np.ones should have
            
            """STEP 5.4.1: joint log probs derivative (jlpd) parts 1 and 2 
            are obtained and combined to update the model specific joint
            log probs derivative. The signs carry over from the gradients of
            the predictive and the joint log probs"""
            
            """(dP(y_t|r_t-1, y_1:t-1)*P(r_t-1, y_1:t-1)*P(r_t|r_t-1)), for 
            each r_t and each param, so size is num_params x run_length_num"""
            jlpd_part1_val = (
                gradients_log_predictive_val[:,1:] +
                model_specific_joint_log_probs + 
                np.log(1-cp_model.hazard_vector(1, t))
                ) 
            jlpd_part1_sign = gradients_log_predictive_sign[:,1:]
    
            """(dP(r_t, y_1:t-1)*P(r_t|r_t-1)*P(y_t|r_t-1, y_1:t-1)), for 
            each r_t and each param, so size is num_params x run_length_num"""
            jlpd_part2_val = (
                self.model_specific_joint_log_probabilities_derivative +
                self.r0_log_prob +
                np.log(1-cp_model.hazard_vector(1, t))
                )
            jlpd_part2_sign = (
                self.model_specific_joint_log_probabilities_derivative_sign)
            
            """Combine jlpd parts 1 and 2, store result in dP(r_t, y_1:t-1), 
            the gradient of the joint probabilities, as before the
            size is num_params x run_length_num"""
#            print("jlpd_part1_val", jlpd_part1_val.shape)
#            print("jlpd_part2_val", jlpd_part2_val.shape)
#            print("jlpd_part1_sign", jlpd_part1_sign.shape)
#            print("jlpd_part2_sign", jlpd_part2_sign.shape)
#            print("gradients_log_predictive_val", gradients_log_predictive_val.shape)
#            print("gradients_log_predictive_sign", gradients_log_predictive_sign.shape)
            res_val, res_sign = scipy.misc.logsumexp(
                a = np.array([jlpd_part1_val, jlpd_part2_val]),
                b = np.array([jlpd_part1_sign, jlpd_part2_sign]),
                return_sign=True,
                axis=0
                )
#            print("res_val", res_val.shape)
#            print("res_sign", res_sign.shape)
            
            """So far, we have only computed the derivatives for non-CPs. 
            Next, we compute the derivative for a CP, using the recursion:
            CP_grad_1 = sum( P(r=0|r_t-1)*dP(y_t|y_1:t-1)*P(y_1:t-1, r_t-1) ) 
            CP_grad_2 = sum( P(r=0|r_t-1)*P(y_t|y_1:t-1)*dP(y_1:t-1, r_t-1) ).
            Size is num_params x run_length_num
            """
            results_grad_1 = (
                scipy.misc.logsumexp(
                        a = np.array([
                             gradients_log_predictive_val[:,1:] + #[:,np.newaxis] +
                             np.log(cp_model.hazard_vector(1,t)) + 
                             model_specific_joint_log_probs                            
                        ]),
                        b = gradients_log_predictive_sign[:,1:], #[:,np.newaxis],
                        return_sign = True,
                        axis=1
                    )
                )
            CP_grad_1_val, CP_grad_1_sign = (results_grad_1[0].flatten(), 
                                             results_grad_1[1].flatten())
            results_grad_2 = (
                scipy.misc.logsumexp(
                    a = np.array([
                         self.model_specific_joint_log_probabilities_derivative +
                         np.log(cp_model.hazard_vector(1,t)) + 
                         self.one_step_ahead_predictive_log_probs                          
                    ]),
                    b = (
                      self.model_specific_joint_log_probabilities_derivative_sign),
                    return_sign = True,
                    axis=1
                    )
                )
            CP_grad_2_val, CP_grad_2_sign = (results_grad_2[0].flatten(), 
                                             results_grad_2[1].flatten())
            
            CP_grad_val, CP_grad_sign = scipy.misc.logsumexp(
                    a = np.array([CP_grad_1_val, CP_grad_2_val]),
                    b = np.array([CP_grad_1_sign, CP_grad_2_sign]),
                    return_sign = True,
                    axis=1
                    )
            

            
            """Also compute the sum of the gradients for the model
            specific joint probabilities over all run lengths, i.e.
            sum(dP(r_t, y_1:t-1)) (quantity is needed as a 
            constant in later computations), size is num_params x 1"""
            #DEBUG: Only needed for Caron's method
#            sum_jlpd_gradients_val, sum_jlpd_gradients_sign = (
#                scipy.misc.logsumexp(
#                a = self.model_specific_joint_log_probabilities_derivative,
#                b = self.model_specific_joint_log_probabilities_derivative_sign,
#                return_sign=True,
#                axis=1)
#                )
                
            """STEP 5.4.2: Compute logs of Q1, Q2, Q3 to finally obtain the 
            desired num_params x 1 vector log(dP(y_t|y_1:t-1)). Define
            Q1 = 1/P(y_t|y_1:t-1) * sum[dP(y_t|y_1:t-1, r_t) * P(r_t|y_1:t-1)], 
            Q2 = 1/P(y_t|y_1:t-1)*1/sum(P(r_t, y_1:t-1)) * 
                     sum(P(y_t|y_1:t-1, r_t)*dP(r_t, y_1:t-1)
            Q3 = 1/P(y_t|y_1:t-1)*[1/sum(P(r_t, y_1:t-1))]^2*
                    [sum(dP(r_t, y_1:t-1))] * 
                    sum(P(y_t|y_1:t-1, r_t)*P(r_t, y_1:t-1)).
            All Q1, Q2, Q3 have size num_params x 1."""
            #DEBUG: Only needed for Caron's method.
            
#            """Compute P(y_t|y_1:t-1) in log form. Notice that the case that
#            the last CP was at t-1 is the latest we take into account, since
#            the RLD used is P(r_t-1|y_1:t-1)"""
#            log_predictive = scipy.misc.logsumexp(
#                    self.one_step_ahead_predictive_log_probs +
#                    run_length_log_distro
#                )
#            """Compute P(y:1_t-1|m_t=m)"""
#            log_evidence = scipy.misc.logsumexp(model_specific_joint_log_probs)
            
            """Q1 = 1/P(y_t|y_1:t-1) * sum[dP(y_t|y_1:t-1, r_t) * 
                    P(r_t|y_1:t-1)]"""
            #DEBUG: Unclear if run length log distro is as long as gradients
#            print("gradients", gradients_log_predictive_val.shape)
#            print("run length distro", run_length_log_distro.shape)
#            Q1, sign1 = -log_predictive + scipy.misc.logsumexp(
#                    a = (gradients_log_predictive_val[:,1:] +
#                        run_length_log_distro),
#                    b = gradients_log_predictive_sign[:,1:],
#                    return_sign=True,
#                    axis=1
#                )
            
#            """Q2 = 1/P(y_t|y_1:t-1)*1/sum(P(r_t, y_1:t-1)) * 
#                     sum(P(y_t|y_1:t-1, r_t)*dP(r_t, y_1:t-1)"""
#            Q2, sign2 = -log_predictive - log_evidence + scipy.misc.logsumexp(
#                a = np.array(self.model_specific_joint_log_probabilities_derivative+
#                    self.one_step_ahead_predictive_log_probs.reshape(
#                            run_length_num)),
#                b = self.model_specific_joint_log_probabilities_derivative_sign,
#                return_sign=True,
#                axis=1
#                )
            
#            """Q3 = 1/P(y_t|y_1:t-1)*[1/sum(P(r_t, y_1:t-1))]^2*
#                    [sum(dP(r_t, y_1:t-1))] * 
#                    sum(P(y_t|y_1:t-1, r_t)*P(r_t, y_1:t-1)).
#                    NOTE: sign3 = (-1)*sum_jlpd_gradients_sign because Q3 is
#                          subtracted, i.e. Q = Q1 + Q2 - Q3. By changing the
#                          sign to -1, we obtain Q = Q1+Q2+Q3."""
##            print("sum_jlpd_gradients_val", sum_jlpd_gradients_val.shape )
#            Q3 = (-log_predictive -2*log_evidence + 
#                         sum_jlpd_gradients_val+
#                         scipy.misc.logsumexp(
#                            a = model_specific_joint_log_probs+
#                                self.one_step_ahead_predictive_log_probs.
#                                reshape(run_length_num))
#                            ) 
#            sign3 = (-1)*sum_jlpd_gradients_sign
            
#            print("Q1", Q1.shape)
#            print("Q2", Q2.shape)
#            print("Q3", Q3.shape)
#            print("sign1", sign1.shape)
#            print("sign2", sign2.shape)
#            print("sign3", sign3.shape)
#            """Get Q and its sign, and finally convert from log-format"""
#            Q, sign = scipy.misc.logsumexp(
#                a=np.array([Q1.reshape(num_params), Q2.reshape(num_params),
#                    Q3.reshape(num_params)]), 
#                b=np.array([sign1.reshape(num_params), 
#                    sign2.reshape(num_params), sign3.reshape(num_params)]),
#                return_sign=True,
#                axis=0)
            
            """STEP 6: Update the derivatives of the joint log probs. The sum
            of these quantities is d/dtheta P(y_1:t), i.e. the gradient for the
            method of Turner et al. Needs to be done this late because the
            d/dtheta P(r_t-1, y_1:t-1) needed for Caron's gradient descent"""
#            print("gradients_log_predictive_val", 
#                  gradients_log_predictive_val.shape)
#            print("self.gradients_log_predictive_sign", 
#                  gradients_log_predictive_sign.shape)
#            print("CP_grad_val_1", CP_grad_1_val.shape)
#            print("CP_grad_val_2", CP_grad_2_val.shape)
#            print("CP_grad_val", CP_grad_val.shape)
            """Compute P(y:1_t-1|m_t=m)"""
            log_evidence = scipy.misc.logsumexp(model_specific_joint_log_probs)
            """Update derivatives"""
            self.model_specific_joint_log_probabilities_derivative = np.insert(
                    res_val, 0, CP_grad_val, axis=1)
            self.model_specific_joint_log_probabilities_derivative_sign = (
                    np.insert(res_sign, 0, CP_grad_sign, axis=1))
            
#            print("Q", Q)
#            print("sign", sign)
#            caron_gradient = sign*np.exp(Q)
            #DEBUG: Check if caron_gradient can be computed more easily as
            #       Turner's gradient, i.e. by summing up the model specific 
            #       lob probs!
            
            #DEBUG: ONLY do this if we have caron's method, since we
            #       might want to have hyperparameter-opt, but with Turner's
            #       method.
            if (self.hyperparameter_optimization == "caron" or 
                self.hyperparameter_optimization == "online"):
                sign, caron_gradient = scipy.misc.logsumexp(
                    a=(self.model_specific_joint_log_probabilities_derivative),
                    b=self.model_specific_joint_log_probabilities_derivative_sign,
                    return_sign=True, axis=1)
                
                """Notice: What we use here is that denoting by theta_1:t the
                sequence of hyperparams used at times 1:t, the derivative
                d/dtheta_t P(y_t|y_1:t-1) = d/dtheta_t {P(y_1:t)/P(y:1:t-1)}, 
                and moreover P(y:1:t-1) does not depend on theta_t! So
                d/dtheta_t P(y_1:t)/P(y_1:t-1)=d/dtheta_t{P(y_1:t)}/P(y_1:t-1),
                which means that in log-scale, we have
                log(d/dtheta_tP(y_t|y:1:t-1)) = log(d/dtheta_t{P(y_1:t)}) -
                                                log(P(y_1:t-1))"""
                caron_gradient = np.exp(caron_gradient)*sign - log_evidence
                
                """STEP 5.5: Call the optimization routine that you selected
                for this model. 'increment' is a num_params x 1 vector that gives
                the direction for each hyperparameter"""
                p, C, scale = 1.0005, 1000, 3 #Nile: 1.005, 1
                step_size = self.step_caron_pow2(t, p, C, scale)
                #increment = step_size * caron_gradient
                self.caron_hyperparameter_optimization(t, caron_gradient, 
                                                       step_size)
            

#            print("self.model_specific_joint_log_probabilities_derivative", 
#                  self.model_specific_joint_log_probabilities_derivative.shape)
#            print("self.model_specific_joint_log_probabilities_derivative_sign", 
#                  self.model_specific_joint_log_probabilities_derivative_sign.shape)
            
            
    #@staticmethod
    def step_caron_pow2(self,t, p, C,scale):
        """for any p \in (1, \infinity), and for any finite constant C this 
        sequence is square summable, but sums to infinity, too."""
        if self.has_lags:
            t_ = t-self.lag_length
        else:
            t_ = t
        return scale*C*(1/(t_+C))
#        #if t>30:
#            return C*(1/t_-30)
#        else:
#            return 0
        
    
    
    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t>1!     
    def update_predictive_distributions(self, y, t, r_evaluations):
        """update the distributions giving rise to the predictive probabilities
        in the next step of the algorithm. Could happen computationally (e.g., 
        INLA) or exactly (i.e., with conjugacy)"""                                                              
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS          
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors 
        at time point t, for all possible run-lengths"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS      
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at 
        time point t, for all possible run-lengths"""
        pass
    
    #SHOULD BE IMPLEMENTED IN EACH SUBLCASS
    def prior_log_density(self, y_flat):
        """Computes the log-density of *y_flat* under the prior. Called by 
        the associated Detector object to get the next segment-density 
        component of the model, denoted D(t,q) for q = NIW"""
        pass
    
    
    #CALLS trimmer(kept_run_lengths), which needs to be implemented in each
    #       subclass of probability_model
    def trim_run_length_log_distrbution(self, t, threshold, trim_type):
        """Trim the log run-length distribution of this model such that any 
        run-lengths with probability mass < threshold will be deleted"""
        
        """If we want to apply thresholding, trim the distributions, see 
        STEP 2 and STEP 3. Otherwise, don't do anything in this function 
        apart from keeping track of all the run-lengths"""      
        if (not ((threshold is None) or (threshold == 0) or (threshold == -1))):
            
            """STEP 2: Find out which run-lengths we need to trim now. Note 
            that P(r_t|m_t,y_1:t) = P(r_t,m_t,y_1:t)/P(m_t,y_1:t), and that
            P(m_t,y_1:t) is what I store as model log evidence"""
            run_length_log_distr = (self.joint_log_probabilities - 
                                    self.model_log_evidence)
            
            if trim_type == "threshold":
                """STEP 2A: If we trim using a threshold, trim accordingly"""
                kept_run_lengths = np.array(run_length_log_distr) >= threshold
                
            elif trim_type == "keep_K":
                """STEP 2B: If we trim discarding all but the K largest run
                length probabilities, trim accordingly"""
                K = min(threshold, np.size(run_length_log_distr))
                max_indices = (-run_length_log_distr).argsort()[:K]
                kept_run_lengths = np.full(np.size(run_length_log_distr), 
                                           False, dtype=bool)
                kept_run_lengths[max_indices] = True
                
            elif trim_type == "SRC":
                """STEP 2C: If we keep all particles larger than alpha and 
                resample the remaining N-A ones"""
                
                """STEP 2.C.0: Depending on whether threshold>0 or <0, design
                alpha as threshold or as K-th largest value's value"""
                if threshold < 0: #i.e., we give the log-prob format directly
                    alpha = threshold
                elif threshold > 0: #i.e., we give K max type input
                    #Choose K-th largest particle's weight as alpha
                    K = min(threshold, np.size(run_length_log_distr))
                    alpha_index = (-run_length_log_distr).argsort()[K-1]
                    alpha = run_length_log_distr[alpha_index]
                    
                
                """STEP 2C.1: Keep all particles larger than alpha"""
                kept_run_lengths = np.array(run_length_log_distr) >= alpha
                #A = np.sum(kept_run_lengths, dtype=int)
                #smaller_alpha_indices = #not larger_alpha
                
                """STEP 2C.2: For the remaining particles, apply stratified
                resampling"""
                u = np.random.uniform(low=0, high=alpha)
                for (index, rl_particle) in zip(range(0, np.size(
                        run_length_log_distr)), run_length_log_distr):
                    if rl_particle < alpha:
                        #u = u-rl_particle in non-log format is <= 0 iff
                        #it holds that u<=rl_particle                        
                        if u <= rl_particle:
                            kept_run_lengths[index] = True
                            run_length_log_distr[index] = alpha
                            #DEBUG: log sum exp
                            u = scipy.misc.logsumexp(
                                    np.array([u, alpha, -rl_particle]))                    

            
            """STEP 3: Drop the quantities associate with dropped run-lengths.
            If there is at least one probability kept, do as usual. If the 
            number of probabilities kept is 0, retain r=t-1"""
            """NOTE: Should only be a problem if trim_type is threshold"""
            if trim_type == "SRC" and np.sum(kept_run_lengths)>1:
                #we need to replace the joint_log_probs inside the model object
                #with the new alpha-replaced ones.
                self.joint_log_probabilities = (run_length_log_distr + 
                                   self.model_log_evidence)
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths)>1:
                self.trimmer(kept_run_lengths)
            elif np.sum(kept_run_lengths)<=1:
                #get the two largest entries and only keep them.
                max_indices = (-run_length_log_distr).argsort()[:2]
                kept_run_lengths[max_indices] = True
                self.trimmer(kept_run_lengths)
    
    
    
    
    
    
    
    