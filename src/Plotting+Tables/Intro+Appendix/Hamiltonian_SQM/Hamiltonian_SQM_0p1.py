#This code was last modified on 09/11/2024
#==============================================================================================

# This file is called called Hamiltonian_SQM_0p1.py
# It is available at https://github.com/emanuele-mendicelli/0p1_Supersymmetric_Quantum_Mechanics
#
# The code below is for generating the Hamiltonian for a 0+1 dimensional supersymmetric quantum mechanics.
# The code was used to produce the results in the proceeding: 
# Towards quantum simulation of lower-dimensional supersymmetric lattice models.
#                                 E. Mendicelli and D. Schaich
#                                 http:// [hep-lat] (2024)
# If you find this code (or parts of it) useful, please cite the preprint as well as the code.
# emanuelemendicelli@hotmail.it
#==============================================================================================


#Code Structure:

#1) Importing the needed packages
#2) Definition of the operators p and q 
#3) Definition of the Hamiltonian
#4) Functions needed during the testing mode

#==============================================================================================


# In this block the needed packages are imported 

import numpy as np 
import scipy as sp

#pakages needed 
from scipy.sparse import diags_array
from scipy.sparse.linalg import matrix_power

#==============================================================================================

#In this code we define the hamiltonian for a 0+1 supersymmetric quantum mechanics (SQM) using only sparse array.


#Defining the encoding for the two bosonic operators p and q.

#Defining the truncated version of the position operator (q) and the momentum operator (p) using the standard definition
# from the simple Harmonic Operator in quantum mechanics, allowing only \Lambda bosonic modes.
# (See J. J. Sakurai and Jim Napolitano - Modern Quantum Mechanics-Cambridge University Press_2021, chapter 2 page 86)

#def p and q using the simple harmonic q_sho p_sho

def q_sho(n_bosonic_modes,mass):
       
       #generating the list of diagonal elements [1, sqrt(2),..., sqrt(n_bosonic_modes-1)]
       diagonal_elements =np.array([np.sqrt(i+1) for i in range(n_bosonic_modes-1)])
       
       # Construct the sparse matrix by placing elements along the principal diagonal  and its parallel diagonals. 
       # The 'offsets' correspond to: 0 -> principal diagonal, 1 -> first upper diagonal, -1 -> first lower diagonal, and so on.
       q_sparse_matrix = (1.0/np.sqrt(2*mass)) * diags_array([diagonal_elements, diagonal_elements], offsets=[-1, 1])

       return q_sparse_matrix 
#==============================================================

def p_sho(n_bosonic_modes, mass):
       
       #generating the list of diagonal elements [1, sqrt(2),..., sqrt(n_bosonic_modes-1)]
       diagonal_elements =np.array([np.sqrt(i+1) for i in range(n_bosonic_modes-1)])
       
       # Construct the sparse matrix by placing elements along the principal diagonal  and its parallel diagonals. 
       # The 'offsets' correspond to: 0 -> principal diagonal, 1 -> first upper diagonal, -1 -> first lower diagonal, and so on.
       p_sparse_matrix = 1j*(np.sqrt(mass * 0.5)) * diags_array([diagonal_elements, (-1) * diagonal_elements], offsets=[-1, 1])

       return p_sparse_matrix
#======================================================================================================================================



# Function to calculate the Hamiltonian with a selected superpotential.

# It returns a sparse array, in case the corresponding dense matrix is needed use the the command  Hamiltonian_SQM_0p1(----).toarry()
# If matrix representation is needed, consider using the print_matrix_nicely() function, 
# as its name suggests, it formats the output in a more readable way!

def Hamiltonian_SQM_0p1(potential_name, n_bosonic_modes=1, m=1, g=1, mu=1):

    # potential_name is a text string like "HO", "AHO", "DW"
    # n_bosonic_modes is the number of bosonic modes allowed
    # m, g and mu are the parameters of the potentials 

    #For the moment there is only one possible choice for the discretization of p and q, the one from the
    #simple quantum Harmonic Oscillator
    q = q_sho(n_bosonic_modes,m)
    p = p_sho(n_bosonic_modes,m)


    #Defining the fermionic and bosonic Identity operator:
    #The size of the fermionic Hilber space is fixed (2x2), therefore
    Id_f=sp.sparse.eye(2)

    Z_sparse = sp.sparse.csr_matrix(([1, -1] , ([0, 1], [0, 1])))

    #The size of the bosonic Hilber space depends from the number of bosonic modes (Lambda x Lambda), therefore
    Id_b=sp.sparse.eye(n_bosonic_modes)


    #List of functions defining the Hamiltonian for each considered potential

    #Hamiltonian    Quantum Harmonic Oscillator
    def H_HO():
        Hb_HO = sp.sparse.kron( Id_f, ( matrix_power(p,2)*0.5 + (m**2 * 0.5) * matrix_power(q,2) ) )

        Hf_HO = (m*0.5)* sp.sparse.kron( Z_sparse, Id_b)

        return Hb_HO + Hf_HO
    #================================================================================================

    #Hamiltonian    Double Well
    def H_DW():
        Hb_DW = sp.sparse.kron( Id_f, ( matrix_power(p,2)*0.5 + 0.5*(g**2)*matrix_power(q, 4) +  (m * g)* matrix_power(q,3) + ( m**2 *0.5 + g**2 * mu**2 )* matrix_power(q,2) + m*g*mu**2 * q ))

        Hf_DW = (m*0.5)* sp.sparse.kron( Z_sparse, Id_b)

        Hint_DW = g*sp.sparse.kron(Z_sparse, q)

        Hconst_DW =  ( 0.5 * (g**2) * (mu**4) )* sp.sparse.kron( Id_f, Id_b)
        
        return  Hb_DW + Hf_DW + Hint_DW + Hconst_DW
    #================================================================================================

    #Hamiltonian       Anharmonic Oscillator
    def H_AHO():
        Hb_AHO = sp.sparse.kron( Id_f, ( matrix_power(p,2)*0.5 + ( m**2 *0.5 )* matrix_power(q,2) + ( m*g )* matrix_power(q,4) + ( g**2 * 0.5 )* matrix_power(q,6) ))

        Hf_AHO = (m*0.5)* sp.sparse.kron( Z_sparse, Id_b)

        Hint_AHO = (1.5*g) * sp.sparse.kron(Z_sparse, matrix_power(q, 2))
        
        return  Hb_AHO + Hf_AHO + Hint_AHO
    #================================================================================================

    
    #Dictionary to handle the selection of the Hamiltonian for different superpotentials
    potential_dict = {
            "HO": H_HO(),
            "DW": H_DW(),
            "AHO": H_AHO()
            }

    return potential_dict.get(potential_name, "Potential not present! type " + str([i for i in potential_dict.keys()]))
#===================================================================================



# Some functions needed during the testing mode

#Function to print nicely a matrix (2-dimensional array) that takes into account the leght of the matrix elements
def print_matrix_nicely(matrix):
    
    #Number of decimals retained 
    n_decimals= 4
    
    # Find the maximum width of the elements in the matrix
    max_width = max(len(str(np.round(item,n_decimals))) for row in matrix for item in row)
    
    # Print each row, with each element centered within the maximum width
    for row in matrix:
        print("  ".join(f"{str(np.round(item,n_decimals)).center(max_width)}" for item in row))
#======================================================================================================================================