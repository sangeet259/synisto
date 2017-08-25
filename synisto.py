import numpy as np
import math
import scipy
import scipy.optimize  # for fmin_cg to optimize the cost function
import pickle



# To use off-the-shelf minimizers we need to flatten these matrices
# into one long vector

def flattenParams(myX, myTheta):
    """
    Hand this function an X matrix and a Theta matrix and it will flatten
    it into into one long (nm*nf + nu*nf,1) shaped numpy array
    """
    return np.concatenate((myX.flatten(),myTheta.flatten()))




# A utility function to re-shape the X and Theta will probably come in handy
def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
    assert flattened_XandTheta.shape[0] == int(nm*nf+nu*nf)
    
    reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
    reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
    
    return reX, reTheta




def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    
    # Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
  
    # Note: 
    # X Shape is (nm x nf), Theta shape is (nu x nf), Y and R shape is (nm x nu)
    # Behold! Complete vectorization
    
    # First dot theta and X together such that you get a matrix the same shape as Y
    term1 = myX.dot(myTheta.T)
    
    # Then element-wise multiply that matrix by the R matrix
    # so only terms from movies which that user rated are counted in the cost
    term1 = np.multiply(term1,myR)
    
    # Then subtract the Y- matrix (which has 0 entries for non-rated
    # movies by each user, so no need to multiply that by myR... though, if
    # a user could rate a movie "0 stars" then myY would have to be element-
    # wise multiplied by myR as well) 
    # also square that whole term, sum all elements in the resulting matrix,
    # and multiply by 0.5 to get the cost
    cost = 0.5 * np.sum( np.square(term1-myY) )
    
    # Regularization stuff
    cost += (mylambda/2.) * np.sum(np.square(myTheta))
    cost += (mylambda/2.) * np.sum(np.square(myX))
    
    return cost



# Remember: use the exact same input arguments for gradient function
# as for the cost function (the off-the-shelf minimizer requires this)
def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    
    # Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)

    # First the X gradient term 
    # First dot theta and X together such that you get a matrix the same shape as Y
    term1 = myX.dot(myTheta.T)
    # Then multiply this term by myR to remove any components from movies that
    # weren't rated by that user
    term1 = np.multiply(term1,myR)
    # Now subtract the y matrix (which already has 0 for nonrated movies)
    term1 -= myY
    # Lastly dot this with Theta such that the resulting matrix has the
    # same shape as the X matrix
    Xgrad = term1.dot(myTheta)
    
    # Now the Theta gradient term (reusing the "term1" variable)
    Thetagrad = term1.T.dot(myX)

    # Regularization stuff
    Xgrad += mylambda * myX
    Thetagrad += mylambda * myTheta
    
    return flattenParams(Xgrad, Thetagrad)





def normalizeRatings(myY, myR):

    """
    Preprocess data by subtracting mean rating for every movie (every row)
    This is important because without this, a user who hasn't rated any movies
    will have a predicted score of 0 for every movie, when in reality
    they should have a predicted score of [average score of that movie].
    """

    # The mean is only counting movies that were rated
    sumY = np.sum(myY,axis=1)
    sumR = np.sum(myR,axis=1)
    # tempY = 
    
    Ymean = sumY/sumR
    Ymean = Ymean.reshape((Ymean.shape[0],1))
    
    return myY-Ymean, Ymean    






# All the tesing will be done here
# Problems:
#   1. The NAN problem
#   2. Ratings outside 5 problem , the root cause is Y mean which is being addes.
#       Do we really need it,
#       Because we are not having any new user who hasn't watched a single movie
#   3. Tweak Lambda
#   4. Tweak the number of mathematical genres , ie nf
#   5. Check for bias , variance (plot cost vs number of features graph)
#   6. What's the bias Jeremy Howard was talking about in Lecture 4
#   7. MaxIter :: Use Callback in fmin_cg
#	8. TensorFlow for better results (gardient Descent insated of fmin_cg)
#  
# Best will be to use pandas dataframe to gain some insights into it
# 


#   Convert this numpy array into a Pandas Dataframe
#   










def main():

    with open("data_y","rb") as f:
    Y = pickle.load(f)

    with open("data_R","rb") as f:
    R = pickle.load(f)
    nm, nu = Y.shape
    nf = 10

    # Y is no_moviesxno_users containing ratings (1-5) of no_movies movies on no_users users
    # a rating of 0 means the movie wasn't rated
    # R is no_moviesxno_users containing R(i,j) = 1 if user j gave a rating to movie i

    # The i-th row of X corresponds to the feature vector x(i) for the i-th movie, 
    # and the j-th row of Theta corresponds to one parameter vector θ(j), for the j-th user. 
    # Both x(i) and θ(j) are n-dimensional vectors. For the purposes of this exercise, 
    # you will use n = 100, and therefore, x(i) ∈ R100 and θ(j) ∈ R100. Correspondingly, 
    # X is a nm × 100 matrix and Theta is a nu × 100 matrix.



    # Let's Randomly Initialise X and Theta


    X = np.random.rand(nm,nf)
    Theta = np.random.rand(nu,nf)


    # The "parameters" we are minimizing are both the elements of the
    # X matrix (nm*nf) and of the Theta matrix (nu*nf)
    # Ynorm, Ymean = normalizeRatings(Y,R)


    myflat = flattenParams(X, Theta)

    # Choose your own regulaisation parameter
    mylambda = 10.

    # Training the actual model with fmin_cg
    result = scipy.optimize.fmin_cg(cofiCostFunc, x0=myflat, fprime=cofiGrad,args=(Y,R,nu,nm,nf,mylambda),maxiter=50,disp=True,full_output=True)


    # Save the trained model for future use
    with open("trained","wb") as f:
        pickle.dump(result,f)



    # Reshape the trained output into sensible "X" and "Theta" matrices
    resX, resTheta = reshapeParams(result[0], nm, nu, nf)



    # After training the model, now make recommendations by computing
    # the predictions matrix
    prediction_matrix = resX.dot(resTheta.T)

main()









# # ************************************************************************************************************

# # Grab the last user's predictions (since I put my predictions at the
# # end of the Y matrix, not the front)
# # Add back in the mean movie ratings
# my_predictions = prediction_matrix[:,-1] + Ymean.flatten()

# for i in range(len(my_predictions)):
#     if math.isnan(my_predictions[i]):
#         my_predictions[i] = 0 

# # Sort my predictions from highest to lowest
# pred_idxs_sorted = np.argsort(my_predictions)
# pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

# print ("Top recommendations for you:")
# for i in range(10):
#     print ('Predicting rating %0.1f for movie id %d .' % (my_predictions[pred_idxs_sorted[i]],pred_idxs_sorted[i]))
    

# # print ("\nOriginal ratings provided:")
# # for i in range(len(my_ratings)):
# #     if my_ratings[i] > 0:
# #         print ('Rated %d for movie %s.' % (my_ratings[i],movies[i]))
