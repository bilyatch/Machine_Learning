# Machine learning with scikit-learn keras & tenserflow

I'm gonna learn new thing related to ML and update you guys on a daily note

<i><u><h3>Day one</h2></u></i>
<h2>Fundamental of Machine Learning</h2>
<t> Here I've learned about Machine learning, use of it and types of ML system
<h3>Machine Learning</h3>
Machine learing is the science of programming computer where the machine learns from the data 
 
 <h4>Uses:</h4>
 <ol type="1">
<li>Healthcare and Medical Diagnosis</li>
<li>Commute Predictions</li>
<li>Social Media</li>
<li>Smart Assistants</li>
<li>Optimizing Search Engine Results</li>
<li>Optimizing Search Engine Results</li>
<font size=-2>and many more</font>

<h4>Types of ML</h4>
<ul>
<li>Supervised learning</li>
<li>Unsupervised learning</li>
<li>Semi-supervised learning</li>
<li>Reinforcement Learning</li>
</ul>

<h3>Supervised learning </h3>
<t>A typical supervised learning task is classification. In Supervised learning data is labeled.

Some of the imp. supervised algorithm 
<ul style="list-style-type:disc;">
<li> k-Nearest Neighbors</li>
<li>Linear Regression</li>
<li>Logistic Regression</li>
<li>Support Vector Machines (SVMs)</li>
<li>Decision Trees and Random Forests</li>
<li>Neural networks2</li>
</ul>
 
 <h3>Unsupervised learning</h3>
 Here, Data is not labeled.
 Some of the unsupervised algorithm are:
 <ol>
 <li> Clustering/li>
 <ul>
<li> K-Means</li>
<li> DBSCAN</li>
<li> Hierarchical Cluster Analysis (HCA)</li>
  </ul>
<li>Anomaly detection and novelty detection</li>
   <ul>
<li>One-class SVM</li>
<li>Isolation Forest</li>
  </ul>
<li>Visualization and dimensionality reduction</li>
     <ul>
<li>Principal Component Analysis (PCA)</li>
<li>Kernel PCA</li>
<li>Locally-Linear Embedding (LLE)</li>
<li>t-distributed Stochastic Neighbor Embedding (t-SNE)</li>
<li>Association rule learning</li>
       </ul>
       <ul>
<li>Aprior</li>
         </ul>
 
 <h3>Semi-supervised learning</h3>
  In this learning algorithm we uses little bit of labeled and alot of unlabeled data
  
  <h3>Reinforecement learning</h3>
  This learning algorithm observe the environment,select and perform actions, get reward in return or get negative reward. This learing algorithm should learn itself what's the best strategy for the problem which is called policy.
  
  
  
  <u>
  <i><h3>Day two </h3></i>
  </u>
  <table>
  <tr>
    <th>Batch learning </th>
    <th>Online learning </th>
  </tr>
  <tr>
    <td> In batch learning, data is accumulated over a period of time and the machine learning model is then trained with this accumulated data from time to time in batches</td>
    <td>Online learning is a method of machine learning where data is processed in real-time as it arrives</td>
  </tr>
     <tr>
    <td>The model is unable to learn incrementally from a stream of live data</td>
    <td>Online learning algorithms take an initial guess model and then pick up one observation from the training population and recalibrate the weights on each input parameter</td>
  </tr>
  </table>
  
  <br></br>
   <table>
  <tr>
    <th>Instance-based learning </th>
    <th> Model-based learning </th>
  </tr>
  <tr>
   <td> In instance-based learning, also known as memory-based learning, the algorithm memorizes the training examples and uses them to make predictions on new data.</td>
   <td> In contrast, model-based learning aims to learn a generalizable model that can be used to make predictions on new data</td>
    </tr>
    <tr>
     <td>instance-based learning simply memorizes the training examples</td>
     <td> Model-based learning aims to learn a generalizable model that can be used to make predictions on new data</td>
    </tr>
 </table>
 <i>
  <h3>Day three</h3></i>
The main steps that should be followed:
<ol>
<li>Look at the big picture.</li>
<li>Get the data</li>
<li>Discover and visualize the data to gain insights.
</li>
<li>Prepare the data for Machine Learning algorithms.</li>
<li>Select a model and train it.</li>
<li>Fine-tune your model.
</li>
<li>Present your solution.</li>
<li>Launch, monitor, and maintain your system</li>
</ol>

 <h3>Look at the big picture</h3>
At this step we need to know our main objective. We need the frame the problem and by framing it we can classify which types of techniques to follow to tackle our problem.
 
 Later we need to select a performance measure
A typical performance measure for
regression problems is the Root Mean Square Error (RMSE) which can be compute as:
 ![image](https://user-images.githubusercontent.com/104844487/235459949-a2102f52-d7d3-4ec8-a389-6173bade4cde.png)
 here,
 <ul>
  <li>m is the number of instances in the dataset you are measuring the RMSE on.</li>
  <li>x<sup>(i)</sup> is a vector of all the feature values of the i<sup>th</sup> instance in the dataset, and y<sup>(i)</sup> is its label 
   <li>RMSE(X,h) is the cost function measured on the set of examples using your
    hypothesis h.</li>
 </ul>
   
Even though the RMSE is generally the preferred performance measure for regression
tasks, in some contexts you may prefer to use another function i.e Mean Absolute Error( also called the Average Absolute Deviation)
![image](https://user-images.githubusercontent.com/104844487/235461861-db5c376a-cfe7-457b-9702-9dff8c83b4a1.png)
 
 Both RMSE and MAE measure the distance between two vectors that is vector of prediction and vector of target values.
 
 <h3>Gradient Descent</h3>
 Gradient descent is used all over the place in machine learning, not just for linear regression, but for training for example some of the most advanced neural network models.  gradient descent is an algorithm that you can use to try to minimize any function.<br>
(https://github.com/bilyatch/Machine_Learning/blob/main/photos/reduced%20funciton.png?raw=true)<br>
 
 <h3>Gradient descent algorithm<h3><br>
![image](https://github.com/bilyatch/Machine_Learning/blob/main/photos/gradientdescent_algo.PNG)
Here,Alpha is also called the learning rate. The learning rate is usually a small positive number between 0 and 1 and it might be say, 0.01. What Alpha does is, it basically controls how big of a step you take downhill. 

<i><h3>Day four</h3></i>
<h3>Training Linear Regression</h3><br>
![image](https://user-images.githubusercontent.com/104844487/235959464-5ccf283f-4460-41a0-9465-85132cf8e9c3.png)<br>
here when derivative term is positive number then,
w=w-α.(positive no.),which decreases the value of w 
but, if the derivative term is negative then,
w=w-α.(negative no.),which increases the value of w.

<h3>Gradient descent for linear alogorithm</h3>
![image](https://user-images.githubusercontent.com/104844487/236116086-9357383c-b643-43f1-ad90-e30fa83e8d97.png)
In Cost Function,depending on where you initialize the parameters w and b, you can end up at different local minima.
In Squared Error Cost Function,the cost function does not and will never have multiple local minima. It has a single global minimum because of this bowl-shape.It's a convex function. Which is as show in fig given below:
![image](https://user-images.githubusercontent.com/104844487/236116595-1998f177-44ab-456a-a9c1-a5a9a45c6063.png)
  
  <i><h3>Day five</h3></i>
  <h4>Multi features</h4>
 Linear regression that look at not just one feature, but a lot of different features.<br>
![image](https://user-images.githubusercontent.com/104844487/236664860-f8156105-8143-489e-a7fc-c7fac9005428.png)<br>
 The name for this type of linear regression model with multiple input features is multiple linear regression.<br>
  
  <h4>Vectorization</h4>
 When you're implementing a learning algorithm, using vectorization will both make your code shorter and also make it run much more efficiently. NumPy dot function is a vectorized implementation of the dot product operation between two vectors and especially when n is large, this will run much faster that's written as :<br>
<t>  f=np.dot(w,x)+b <br>
 
  


  



 
 

 
