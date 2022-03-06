#!/usr/bin/env python
# coding: utf-8

# # CE-40717: Machine Learning

# ## HW8-Clustering & Reinforcement Learning
# 
# Amir Pourmand - 99210259

# ### Kmeans & GMM:
# 
# At this question, we tend to implement Kmeans & GMM algorithms. For this purpose, `DO NOT EMPLOY` ready-for-use python libraries. Use this implementation for solving the following questions. Kmeans should continue till centeroids won't change. Furthermore, GMM also should continue till the difference of two consecutive likelihood logarithm would be less than 0.1. Notice that after executing the Kmeans part, the primitive centroids of GMM should be identical with ultimate Kmeans centroids.

# In[8]:


from sklearn.datasets.samples_generator import make_classification, make_moons, make_circles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# #### Part 1:
# 
# Utilize the subsequent cell in order to create the Dataset. Afterwards, try to execute the algorithm with k=2 centroids. At Kmeans, it is recommended to execute the algorithm with several various starting states in order to eventually choose the best respective result.

# In[9]:


X,Y = make_classification(n_samples=700, n_features=10, n_informative=5,
                          n_redundant=0, n_clusters_per_class=2, n_classes=3)


# ## KMeans Implementation

# In[10]:


class KMeans:

    def __init__(self, n_clusters = 3, tolerance = 0.01, max_iter = 100, runs = 1):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter
        self.runs = runs
        
    def fit(self, X,Y):
        row_count, col_count = X.shape
        
        X_values = self.__get_values(X)
        
        X_labels = np.zeros(row_count)
        
        costs = np.zeros(self.runs)
        all_clusterings = []

        for i in range(self.runs):
            cluster_means =  self.__initialize_means(X_values, row_count)

            for _ in range(self.max_iter):            
                previous_means = np.copy(cluster_means)
                
                distances = self.__compute_distances(X_values, cluster_means, row_count)
            
                X_labels = self.__label_examples(distances)
            
                cluster_means = self.__compute_means(X_values, X_labels, col_count)

                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break
            
            X_values_with_labels = np.append(X_values, X_labels[:, np.newaxis], axis = 1)
            
            all_clusterings.append( (cluster_means, X_values_with_labels) )
            costs[i] = self.__compute_cost(X_values, X_labels, cluster_means)
        
        best_clustering_index = costs.argmin()
        
        self.costs = costs
        self.cost_ = costs[best_clustering_index]
        
        self.centroid,self.items = all_clusterings[best_clustering_index]
        self.y = Y
        return all_clusterings[best_clustering_index]
        
    def __initialize_means(self, X, row_count):
        return X [ np.random.choice(row_count, size=self.n_clusters, replace=False) ]
        
        
    def __compute_distances(self, X, cluster_means, row_count):
        distances = np.zeros((row_count, self.n_clusters))
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis = 1)
            
        return distances
    
    def __label_examples(self, distances):
        return distances.argmin(axis = 1)
    
    def __compute_means(self, X, labels, col_count):
        cluster_means = np.zeros((self.n_clusters, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            if len(cluster_elements):
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis = 0)
                
        return cluster_means
    
    def __compute_cost(self, X, labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X [ labels == cluster_mean_index ]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis = 1).sum()
        
        return cost
            
    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)
    
    def predict(self):
        data=pd.DataFrame(self.items)
        added_column=list(data.columns)[-1]
        data['Label'] = self.y
        resultOfClustering=data.groupby([added_column])['Label'].agg(lambda x: x.value_counts().index[0])
        mapping = dict()
        for label in range(self.n_clusters):
            label_predicted = resultOfClustering[label]
            mapping[label] = label_predicted
        data['PredictedLabels']=data[added_column].map(mapping)
        return np.array(data['PredictedLabels'])


# In[11]:


kmeans=KMeans(2,max_iter=10000,runs=20)
centroids,kmeans_items=kmeans.fit(X,Y)
plt.plot(np.arange(len(kmeans.costs)),kmeans.costs)
plt.title('error of different runs')
plt.xticks(np.arange(len(kmeans.costs)))
plt.show();


# In[ ]:





# ## Gaussian Mixture Model Implementation

# In[12]:


import numpy as np

import scipy.stats as sp
class GaussianMixModel():
    def __init__(self, X, k=2):
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        self.k = k
        
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k
        self.Z = np.asmatrix(np.empty((self.m, self.k), dtype=float))

     
    def initialize_means(self,means):
        self.mean_arr = means
    
    def fit(self, tol=0.1):

        num_iters = 0
        logl = 1
        previous_logl = 0
        while(logl-previous_logl > tol):
            previous_logl = self.loglikelihood()
            self.e_step()
            self.m_step()
            num_iters += 1
            logl = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, logl))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, logl))

    def loglikelihood(self):
        logl = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                tmp += sp.multivariate_normal.pdf(self.data[i, :],self.mean_arr[j, :].A1,self.sigma_arr[j, :]) * self.phi[j]
            logl += np.log(tmp)
        return logl


    def e_step(self):
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = sp.multivariate_normal.pdf(self.data[i, :],
                                                       self.mean_arr[j].A1,
                                                       self.sigma_arr[j]) *\
                      self.phi[j]
                den += num

                self.Z[i, j] = num
            self.Z[i, :] /= den
            assert self.Z[i, :].sum() - 1 < 1e-4  # Program stop if this condition is false

    def m_step(self):
         for j in range(self.k):
            const = self.Z[:, j].sum()
            self.phi[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.Z[i, j])
                _sigma_j += self.Z[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))

            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
    
    def predict(self):
        return np.array(np.argmax(gmm.Z,axis=1)).flatten()


# In[13]:


gmm=GaussianMixModel(X,k=2)
gmm.initialize_means(np.asmatrix(centroids))
gmm.fit()


# #### Part 2:
# 
# In a separated cell, implement `Purity` and `Rand-Index` criteria in order to compare the performance of mentioned algorithms.

# ## KMeans 

# In[ ]:


print('Purity Of kmeans: ',np.sum(kmeans.predict()==Y)/len(Y))


# In[447]:


from scipy.special import comb
def rand_index_score(clusters, classes):
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = comb(np.bincount(clusters), 2).sum() - tp
    fn = comb(np.bincount(classes), 2).sum() - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


# In[448]:


print('rand index of kmeans', rand_index_score(kmeans.predict(),Y))


# ## Gaussian Mixture Model

# In[449]:


print('purity index: ', np.sum(gmm.predict() == Y)/len(Y))


# In[450]:


print('rand index', rand_index_score(gmm.predict(),Y))


# #### Part 3:
# 
# Use the following cell in order to create new Datasets. Afterwards, try to execute mentioned algorithms on new Dataset and eventually compare the recent results with the help of visualization(there is no problem for using relevant python libraries like `matplotlib`). Consider two clusters for this part.

# In[496]:


X, Y = make_classification(n_samples=700, n_features=2, n_informative=2, n_redundant=0, n_classes=2)


# In[497]:


k=2
kmeans=KMeans(k,max_iter=10000,runs=20)
centroids,kmeans_items=kmeans.fit(X,Y)

color_s =["green","blue","navy","maroon",'orange'] 
for i in range(k):
    plt.scatter(kmeans_items[kmeans_items[:,2]==i,0] , kmeans_items[kmeans_items[:,2]==i,1] 
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(centroids[:,0] , centroids[:,1] , s = 300, color = 'red')
plt.title('Our clusters')
plt.show();


# In[498]:


gmm=GaussianMixModel(X,k)
gmm.initialize_means(np.asmatrix(centroids))
gmm.fit();
gmm_result = gmm.predict()
data=pd.DataFrame(X)
data['Predicted'] = gmm_result

for i in range(k):
    plt.scatter(data[data['Predicted']==i][0], data[data['Predicted']==i][1]
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(np.array(gmm.mean_arr[:,0]).flatten() , np.array(gmm.mean_arr[:,1]).flatten() , s = 300, color = 'red')
plt.show();


# In[499]:


X, Y = make_moons(n_samples=700, noise=0.2)


# In[500]:


k=2
kmeans=KMeans(k,max_iter=10000,runs=20)
centroids,kmeans_items=kmeans.fit(X,Y)

color_s =["green","blue","navy","maroon",'orange'] 
for i in range(k):
    plt.scatter(kmeans_items[kmeans_items[:,2]==i,0] , kmeans_items[kmeans_items[:,2]==i,1] 
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(centroids[:,0] , centroids[:,1] , s = 300, color = 'red')
plt.title('Our clusters')
plt.show();


# In[501]:


gmm=GaussianMixModel(X,k)
gmm.initialize_means(np.asmatrix(centroids))
gmm.fit();
gmm_result = gmm.predict()
data=pd.DataFrame(X)
data['Predicted'] = gmm_result

for i in range(k):
    plt.scatter(data[data['Predicted']==i][0], data[data['Predicted']==i][1]
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(np.array(gmm.mean_arr[:,0]).flatten() , np.array(gmm.mean_arr[:,1]).flatten() , s = 300, color = 'red')
plt.show();


# In[505]:


X, Y = make_circles(n_samples=700, noise=0.2)


# In[506]:


k=2
kmeans=KMeans(k,max_iter=10000,runs=20)
centroids,kmeans_items=kmeans.fit(X,Y)

color_s =["green","blue","navy","maroon",'orange'] 
for i in range(k):
    plt.scatter(kmeans_items[kmeans_items[:,2]==i,0] , kmeans_items[kmeans_items[:,2]==i,1] 
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(centroids[:,0] , centroids[:,1] , s = 300, color = 'red')
plt.title('Our clusters')
plt.show();


# In[507]:


gmm=GaussianMixModel(X,k)
gmm.initialize_means(np.asmatrix(centroids))
gmm.fit();
gmm_result = gmm.predict()
data=pd.DataFrame(X)
data['Predicted'] = gmm_result

for i in range(k):
    plt.scatter(data[data['Predicted']==i][0], data[data['Predicted']==i][1]
                ,s=100, label = "cluster "+str(i), color =color_s[i])

plt.scatter(np.array(gmm.mean_arr[:,0]).flatten() , np.array(gmm.mean_arr[:,1]).flatten() , s = 300, color = 'red')
plt.show();


# ### Reinforcement Learning:
# 
# At the bellow cell, besides the required libraries have been imported, feel free for changing the num_states variable with your desired number.

# In[1]:


import numpy as np
import random
import gym


# In[ ]:


env = gym.make("MountainCar-v0")
num_actions = 3
num_states = 50

# first I should note that first one is position and second one is velocity! 
# so each state should be recognized using two discretized states
q_table = np.zeros(shape=(num_states,num_states, num_actions))

# You may change the inputs of any function as you desire.
SPACE_LOW = env.observation_space.low
SPACE_HIGH = env.observation_space.high
DISCOUNT_FACTOR = 0.95
EXPLORATION = 0.15

EPISODES = 100000
STEP_COUNT_MAX = 20000
DISPLAY=False


# #### Part 1:
# 
# Next cell wants you supplement two functions. First for transforming the continuous space into discrete one (in order to make using q_table feasible), second for updating q_values based on the last action done by agent.

# In[40]:


def discretize_state():
    return np.abs(SPACE_HIGH-SPACE_LOW)/num_states

def env_state_to_Q_state(state):
    return np.round((state - SPACE_LOW)/discretize_state()).astype(int)

#p is position , v is velocity, p_ is position_new, v_ is velocity_new
def update_q(p, v, p_, v_, action, eta, reward):
    if np.random.uniform(0,1) < EXPLORATION:
        action_after = np.random.choice(env.action_space.n)
    else:
        action_after = np.argmax(q_table[p_][v_])
    q_table[p][v][action] = q_table[p][v][action] + eta * (reward + DISCOUNT_FACTOR *  q_table[p_][v_][action_after] - q_table[p][v][action])


# #### Part 2:
# 
# At the following cell, the ends of two functions are getting current action based on the policy and defining the training process respectively.

# In[59]:


# You may change the inputs of any function as you desire.
def get_action():
    global EXPLORATION
    sum_reward_every_thousand = 0
    eta = 0.1
    for episode in range(EPISODES):
        state = env.reset()
        sum_reward = 0
        
        if episode < 30000:
            EXPLORATION = 0.15
            eta = 0.1
        else:
            EXPLORATION = (0.15)* (0.99)**((episode-30000)//100)
            eta = (0.1) * (0.99)**((episode-30000)//10000)


        for step in range(STEP_COUNT_MAX):

            if episode % 1000 == 1 and DISPLAY:
                env.render()

            p, v = env_state_to_Q_state(state)
            if np.random.uniform(0, 1) < EXPLORATION:
                action = np.random.choice(env.action_space.n) 
            else:
                action = np.argmax(q_table[p][v])
            state, reward, done, _ = env.step(action)
            sum_reward += reward

            p_, v_ = env_state_to_Q_state(state)
            update_q(p,v,p_, v_, action, eta, reward)

            if done:
                break
        
        sum_reward_every_thousand+= sum_reward

        if episode % 1000 == 1:
            print(f'Episode: {episode}, Total Reward: {sum_reward}, Mean Reward for previous thousand: {sum_reward_every_thousand/1000}')
            sum_reward_every_thousand=0
        
        

def q_learning():
    return np.argmax(q_table, axis=2) 


def save_policy():
    np.save('policy.npy', q_learning())


# In[60]:


get_action()


# In[61]:


save_policy()


# #### Part 3:
# 
# Ultimately, the score function examines the average performance of Agent (after nearly 1000 times) based on previous implementations.

# In[62]:


# Attention: don't change this function. we will use this to grade your policy which you will hand in with policy.npy
# btw you can use it to see how you are performing. Uncomment two lines which are commented to be able to see what is happening visually.
def score():
    policy, scores = np.load("policy.npy"), []
    for episode in range(1000):
        print(f"******Episode {episode}")
        state, score, done, step = env_state_to_Q_state(env.reset()), 0, False, 0
        while not done:
            # time.sleep(0.04)
            p,v = state
            action = policy[p,v]
            state, reward, done, _ = env.step(action)
            state = env_state_to_Q_state(state)
            step += 1
            score += int(reward)
            
        print(f"Score:{score}")
        scores.append(score)
    print(f"Average score over 1000 run : {np.array(scores).mean()}")

score()


# In[ ]:




