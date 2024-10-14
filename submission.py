import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏󠄉͏󠄈͏︁
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏󠄉͏󠄈͏︁
#
# pgmpy.sampling.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏󠄉͏󠄈͏︁
# pgmpy.factors.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏󠄉͏󠄈͏︁
# pgmpy.estimators.*͏︅͏︀͏︋͏︋͏󠄌͏󠄎͏󠄋͏󠄉͏󠄈͏︁

def make_security_system_net():
    """
    Create a Bayes Net representation of the security system problem.
    Use the following as the name attributes: "H", "C", "M", "B", "Q", "K", "D".
    """
    BayesNet = BayesianNetwork()
    # Add nodes
    BayesNet.add_node("H")  # Spectre hires professional hackers
    BayesNet.add_node("C")  # Spectre buys Contra
    BayesNet.add_node("M")  # Spectre hires mercenaries
    BayesNet.add_node("B")  # Bond is guarding M
    BayesNet.add_node("Q")  # Q’s database is hacked
    BayesNet.add_node("K")  # M gets kidnapped
    BayesNet.add_node("D")  # Spectre obtains the Double-0 files

    # Add edges representing dependencies
    BayesNet.add_edge("H", "Q")  # H influences Q
    BayesNet.add_edge("C", "Q")  # C influences Q
    BayesNet.add_edge("M", "K")  # M influences K
    BayesNet.add_edge("B", "K")  # B influences K
    BayesNet.add_edge("Q", "D")  # Q influences D
    BayesNet.add_edge("K", "D")  # K influences D
    return BayesNet


def set_probability(bayes_net):
    """
    Set probability distribution for each node in the security system.
    Use the following as the name attributes: "H","C", "M","B", "Q", 'K', "D".
    """

    # Prior probabilities for root nodes
    cpd_H = TabularCPD(variable='H', variable_card=2, values=[[0.5], [0.5]])
    cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.7], [0.3]])
    cpd_M = TabularCPD(variable='M', variable_card=2, values=[[0.2], [0.8]])
    cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.5], [0.5]])

    # Conditional probabilities for Q given H and C
    cpd_Q = TabularCPD(
        variable='Q', variable_card=2,
        values=[
            [0.95, 0.75, 0.45, 0.1],  # P(Q=False)
            [0.05, 0.25, 0.55, 0.9]   # P(Q=True)
        ],
        evidence=['H', 'C'],
        evidence_card=[2, 2]
    )

    # Conditional probabilities for K given M and B
    cpd_K = TabularCPD(
        variable='K', variable_card=2,
        values=[
            [0.25, 0.99, 0.05, 0.85],  # P(K=False)
            [0.75, 0.01, 0.95, 0.15]   # P(K=True)
        ],
        evidence=['M', 'B'],
        evidence_card=[2, 2]
    )

    # Conditional probabilities for D given Q and K
    cpd_D = TabularCPD(
        variable='D', variable_card=2,
        values=[
            [0.98, 0.65, 0.40, 0.01],  # P(D=False)
            [0.02, 0.35, 0.60, 0.99]   # P(D=True)
        ],
        evidence=['Q', 'K'],
        evidence_card=[2, 2]
    )

    # Add all CPDs to the Bayesian network
    bayes_net.add_cpds(cpd_H, cpd_C, cpd_M, cpd_B, cpd_Q, cpd_K, cpd_D)

    # Optional: Check if the model is valid
    # assert bayes_net.check_model()

    return bayes_net



def get_marginal_double0(bayes_net):
    """
    Calculate the marginal probability that Double-0 gets compromised.
    """

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    prob = marginal_prob['D'].values
    double0_prob = prob[1]
    return double0_prob

def get_conditional_double0_given_no_contra(bayes_net):
    """
    Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'], evidence={'C': 0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob

def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """
    Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(
        variables=['D'],
        evidence={'C': 0, 'B': 1},
        joint=False
    )
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob

def get_game_network():
    """
    Create a Bayes Net representation of the game problem.
    Name the nodes as "A", "B", "C", "AvB", "BvC", and "CvA".
    """
    # Initialize the Bayesian Network
    BayesNet = BayesianNetwork()

    # Add nodes for skill levels (A, B, C) and match outcomes (AvB, BvC, CvA)
    BayesNet.add_node("A")  # Skill level of Team A (Airheads)
    BayesNet.add_node("B")  # Skill level of Team B (Buffoons)
    BayesNet.add_node("C")  # Skill level of Team C (Clods)
    BayesNet.add_node("AvB")  # Outcome of match between A and B
    BayesNet.add_node("BvC")  # Outcome of match between B and C
    BayesNet.add_node("CvA")  # Outcome of match between C and A

    # Add edges representing dependencies between skill levels and match outcomes
    BayesNet.add_edge("A", "AvB")  # A's skill influences the outcome of A vs B
    BayesNet.add_edge("B", "AvB")  # B's skill influences the outcome of A vs B
    BayesNet.add_edge("B", "BvC")  # B's skill influences the outcome of B vs C
    BayesNet.add_edge("C", "BvC")  # C's skill influences the outcome of B vs C
    BayesNet.add_edge("C", "CvA")  # C's skill influences the outcome of C vs A
    BayesNet.add_edge("A", "CvA")  # A's skill influences the outcome of C vs A

    # Define CPDs for skill levels (A, B, C)
    cpd_A = TabularCPD(variable='A', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD(variable='B', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD(variable='C', variable_card=4, values=[[0.15], [0.45], [0.30], [0.10]])

    # Define CPDs for match outcomes based on skill levels
    # The probabilities are based on the difference in skill levels between the teams
    cpd_AvB = TabularCPD(variable='AvB', variable_card=3,
                         values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],  
                                 [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],  
                                 [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],  
                         evidence=['A', 'B'], evidence_card=[4, 4])

    cpd_BvC = TabularCPD(variable='BvC', variable_card=3,
                         values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],  
                                 [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],  
                                 [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], 
                         evidence=['B', 'C'], evidence_card=[4, 4])

    cpd_CvA = TabularCPD(variable='CvA', variable_card=3,
                         values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],  
                                 [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],  
                                 [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], 
                         evidence=['C', 'A'], evidence_card=[4, 4])

    # Add CPDs to the Bayesian Network
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)

    # Validate the model to make sure it is correct
    assert BayesNet.check_model()

    # Return the created Bayesian Network
    return BayesNet

def calculate_posterior(bayes_net):
    """
    Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss, and tie likelihood.
    """
    # Create a VariableElimination object
    solver = VariableElimination(bayes_net)
    
    # Evidence: A beats B (AvB = 0) and A draws with C (CvA = 2)
    evidence = {'AvB': 0, 'CvA': 2}
    
    # Perform inference to get the posterior distribution of BvC
    posterior_dist = solver.query(variables=['BvC'], evidence=evidence, joint=False)
    
    # Access the values for the posterior distribution
    prob = posterior_dist['BvC'].values  # Extract the values for BvC's possible outcomes
    
    return prob 



def Gibbs_sampler(bayes_net, initial_state):
    """
    Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """

    # Handle invalid input
    if bayes_net is None:
        return None

    if initial_state is None or len(initial_state) == 0:
        initial_state = (
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 3),
            0,
            random.randint(0, 2),
            2
        )
        return initial_state

    # pick variable to sample
    variable_index = random.randint(0, 5)

    # Sample the chosen variable
    sample = tuple(initial_state)

    # A
    if variable_index == 0:
        A_cpd = bayes_net.get_cpds('A').values
        B_value = initial_state[1]
        C_value = initial_state[2]
        AvB_cpd = bayes_net.get_cpds('AvB').values[0]
        CvA_cpd = bayes_net.get_cpds('CvA').values[2]

        likelihood_numerator_A = []
        # Find numerator for posterior calculation
        for i in range(len(A_cpd)):
            numerator = A_cpd[i] * AvB_cpd[i][B_value] * CvA_cpd[C_value][i]
            likelihood_numerator_A.append(numerator)

        # normalize numerators
        sum_A = sum(likelihood_numerator_A)
        likelihoods = np.array(likelihood_numerator_A) / sum_A
        # Randomly select the new value based on the given distribution
        new_A = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = new_A, B_value, C_value, 0, initial_state[4], 2

    # B
    elif variable_index == 1:
        B_cpd = bayes_net.get_cpds('B').values
        A_value = initial_state[0]
        C_value = initial_state[2]
        AvB_cpd = bayes_net.get_cpds('AvB').values[0]
        BvC_cpd = bayes_net.get_cpds('BvC').values[initial_state[4]]

        likelihood_numerator_B = []
        # Find numerator for posterior calculation
        for i in range(len(B_cpd)):
            numerator = B_cpd[i] * AvB_cpd[A_value][i] * BvC_cpd[i][C_value]
            likelihood_numerator_B.append(numerator)

        # normalize numerators
        sum_B = sum(likelihood_numerator_B)
        likelihoods = np.array(likelihood_numerator_B) / sum_B
        # Randomly select the new value based on the given distribution
        new_B = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = A_value, new_B, C_value, 0, initial_state[4], 2

    # C
    elif variable_index == 2:
        C_cpd = bayes_net.get_cpds('C').values
        B_value = initial_state[1]
        A_value = initial_state[0]
        BvC_cpd = bayes_net.get_cpds('BvC').values[initial_state[4]]
        CvA_cpd = bayes_net.get_cpds('CvA').values[2]

        likelihood_numerator_C = []
        # Find numerator for posterior calculation
        for i in range(len(C_cpd)):
            numerator = C_cpd[i] * BvC_cpd[B_value][i] * CvA_cpd[i][A_value]
            likelihood_numerator_C.append(numerator)

        # normalize numerators
        sum_C = sum(likelihood_numerator_C)
        likelihoods = np.array(likelihood_numerator_C) / sum_C
        # Randomly select the new value based on the given distribution
        new_C = np.random.choice([0, 1, 2, 3], p=likelihoods)
        sample = A_value, B_value, new_C, 0, initial_state[4], 2

    # AvB
    elif variable_index == 3:
        AvB_cpd = bayes_net.get_cpds('AvB').values
        A_value = initial_state[1]
        B_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = AvB_cpd[i][A_value][B_value]
            numerators.append(numerator)

        # normalize numerators
        sum_AvB = sum(numerators)
        likelihoods = np.array(numerators) / sum_AvB
        # Randomly select the new value based on the given distribution
        new_AvB = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], A_value, B_value, 0, new_AvB, 2

    # BvC
    elif variable_index == 4:
        BvC_cpd = bayes_net.get_cpds('BvC').values
        B_value = initial_state[1]
        C_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = BvC_cpd[i][B_value][C_value]
            numerators.append(numerator)

        # normalize numerators
        sum_BvC = sum(numerators)
        likelihoods = np.array(numerators) / sum_BvC
        # Randomly select the new value based on the given distribution
        new_BvC = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], B_value, C_value, 0, new_BvC, 2

    # CvA
    elif variable_index == 5:
        CvA_cpd = bayes_net.get_cpds('CvA').values
        C_value = initial_state[1]
        A_value = initial_state[2]

        numerators = []
        for i in range(3):
            numerator = CvA_cpd[i][C_value][A_value]
            numerators.append(numerator)

        # normalize numerators
        sum_CvA = sum(numerators)
        likelihoods = np.array(numerators) / sum_CvA
        # Randomly select the new value based on the given distribution
        new_CvA = np.random.choice([0, 1, 2], p=likelihoods)
        sample = initial_state[0], C_value, A_value, 0, new_CvA, 2

    return sample



def MH_sampler(bayes_net, initial_state):
    """
    Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    # If no initial state, generate one randomly
    if not initial_state:
        initial_state = [
            random.randint(0, 3),  # Skill of A
            random.randint(0, 3),  # Skill of B
            random.randint(0, 3),  # Skill of C
            random.randint(0, 2),  # Result of AvB
            random.randint(0, 2),  # Result of BvC
            random.randint(0, 2),  # Result of CvA
        ]
    
    # Convert initial state to a tuple
    current_state = tuple(initial_state)
    
    # Generate a completely new candidate state (not just a small modification)
    new_a = random.randint(0, 3)
    new_b = random.randint(0, 3)
    new_c = random.randint(0, 3)
    new_avb = random.randint(0, 2)
    new_bvc = random.randint(0, 2)
    new_cva = random.randint(0, 2)
    candidate_state = (new_a, new_b, new_c, new_avb, new_bvc, new_cva)
    
    # Access CPDs for A, B, C and match results
    A_cpd = bayes_net.get_cpds('A').values
    B_cpd = bayes_net.get_cpds('B').values
    C_cpd = bayes_net.get_cpds('C').values
    AvB_cpd = bayes_net.get_cpds('AvB').values
    BvC_cpd = bayes_net.get_cpds('BvC').values
    CvA_cpd = bayes_net.get_cpds('CvA').values
    
    # Calculate the probability of current and candidate states
    def calculate_prob(state):
        A_skill, B_skill, C_skill = state[0], state[1], state[2]
        AvB_result, BvC_result, CvA_result = state[3], state[4], state[5]
        
        # Ensure that the indices are within bounds for the CPDs
        if A_skill < 0 or A_skill > 3 or B_skill < 0 or B_skill > 3 or C_skill < 0 or C_skill > 3:
            return 0  # Invalid state
        
        # Probability of skills
        prob_A = A_cpd[A_skill]
        prob_B = B_cpd[B_skill]
        prob_C = C_cpd[C_skill]
        
        # Probability of match results based on skills
        try:
            prob_AvB = AvB_cpd[A_skill, B_skill][AvB_result]
            prob_BvC = BvC_cpd[B_skill, C_skill][BvC_result]
            prob_CvA = CvA_cpd[C_skill, A_skill][CvA_result]
        except IndexError:
            return 0  # Return 0 probability if there's an indexing error
        
        # Return the total probability as the product of all probabilities
        return prob_A * prob_B * prob_C * prob_AvB * prob_BvC * prob_CvA
    
    current_prob = calculate_prob(current_state)
    candidate_prob = calculate_prob(candidate_state)
    
    # Calculate acceptance probability
    if current_prob == 0:  # Avoid division by zero
        acceptance_prob = 1
    else:
        acceptance_prob = min(1, candidate_prob / current_prob)
    
    # Decide whether to accept the candidate state
    if random.uniform(0, 1) < acceptance_prob:
        return candidate_state  # Accept candidate
    else:
        return current_state  # Keep current state


def compare_sampling(bayesian_network, initial_state):
    """
    Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge.
    """
    gibbs_iterations = 0
    mh_iterations = 0
    mh_rejections = 0
    gibbs_result_distribution = [0, 0, 0]  # Posterior distribution of the BvC match as produced by Gibbs
    mh_result_distribution = [0, 0, 0]  # Posterior distribution of the BvC match as produced by MH

    if bayesian_network is None:
        return None

    # Initialize convergence parameters
    delta_threshold = 0.00001
    max_iterations = 100000

    # Gibbs sampling
    current_distribution = np.array([0, 0, 0])
    previous_distribution = np.array([0, 0, 0])
    current_state = initial_state
    gibbs_convergence_counter = 0
    convergence_limit = 100

    for _ in range(max_iterations):
        new_state = Gibbs_sampler(bayesian_network, current_state)
        current_state = new_state
        current_distribution[new_state[4]] += 1

        # Normalize current and previous distributions to get probabilities
        normalized_current_dist = current_distribution / np.sum(current_distribution)
        normalized_previous_dist = previous_distribution
        if np.sum(previous_distribution) != 0:
            normalized_previous_dist = previous_distribution / np.sum(previous_distribution)

        difference = np.average(np.abs(normalized_current_dist - normalized_previous_dist))

        if difference <= delta_threshold:
            gibbs_convergence_counter += 1
            if gibbs_convergence_counter == convergence_limit:
                gibbs_iterations += 1
                break
        else:
            gibbs_convergence_counter = 0

        previous_distribution = np.copy(current_distribution)
        gibbs_iterations += 1

    gibbs_result_distribution = current_distribution / np.sum(current_distribution)

    # Metropolis-Hastings sampling
    current_distribution = np.array([0, 0, 0])
    previous_distribution = np.array([0, 0, 0])
    current_state = initial_state
    mh_convergence_counter = 0

    for _ in range(max_iterations):
        candidate_state = MH_sampler(bayesian_network, current_state)

        if candidate_state == current_state:
            mh_rejections += 1
        current_distribution[candidate_state[4]] += 1
        current_state = candidate_state

        # Normalize current and previous distributions to get probabilities
        normalized_current_dist = current_distribution / np.sum(current_distribution)
        normalized_previous_dist = previous_distribution
        if np.sum(previous_distribution) != 0:
            normalized_previous_dist = previous_distribution / np.sum(previous_distribution)

        difference = np.average(np.abs(normalized_current_dist - normalized_previous_dist))

        if difference <= delta_threshold:
            mh_convergence_counter += 1
            if mh_convergence_counter == convergence_limit:
                mh_iterations += 1
                break
        else:
            mh_convergence_counter = 0

        previous_distribution = np.copy(current_distribution)
        mh_iterations += 1

    mh_result_distribution = current_distribution / np.sum(current_distribution)

    return gibbs_result_distribution, mh_result_distribution, gibbs_iterations, mh_iterations, mh_rejections


def sampling_question():
    """Question about sampling performance."""
    bayesian_network = get_game_network()
    gibbs_result, mh_result, gibbs_iterations, mh_iterations, mh_rejections = compare_sampling(bayesian_network, [])

    # Assign value to choice and factor
    if gibbs_iterations < mh_iterations:
        best_method_index = 0
        convergence_factor = mh_iterations / gibbs_iterations
    else:
        best_method_index = 1
        convergence_factor = gibbs_iterations / mh_iterations

    sampling_methods = ['Gibbs', 'Metropolis-Hastings']
    return sampling_methods[best_method_index], convergence_factor


def return_your_name():
    """
        Return your name from this function
    """
    return "Erwei Yao"
