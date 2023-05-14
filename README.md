# Machine learning with scikit-learn keras & tenserflow

I'm gonna learn new thing related to ML and update you guys on a daily note

<i><b><h3>Day one</h2></b></i>
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
   <i><b><h3>Day two </h3></b></i>
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
 
 
  <i><b><h3>Day three</h3></b></i>
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
regression problems is the Root Mean Square Error (RMSE) which can be compute as:<br>
 <img src="https://user-images.githubusercontent.com/104844487/235459949-a2102f52-d7d3-4ec8-a389-6173bade4cde.png"><br>
 here,
 <ul>
  <li>m is the number of instances in the dataset you are measuring the RMSE on.</li>
  <li>x<sup>(i)</sup> is a vector of all the feature values of the i<sup>th</sup> instance in the dataset, and y<sup>(i)</sup> is its label 
   <li>RMSE(X,h) is the cost function measured on the set of examples using your
    hypothesis h.</li>
 </ul>
   
Even though the RMSE is generally the preferred performance measure for regression
tasks, in some contexts you may prefer to use another function i.e Mean Absolute Error( also called the Average Absolute Deviation)<br>
 <img src="https://user-images.githubusercontent.com/104844487/235461861-db5c376a-cfe7-457b-9702-9dff8c83b4a1.png"><br>
 
 Both RMSE and MAE measure the distance between two vectors that is vector of prediction and vector of target values.
 
 <h3>Gradient Descent</h3>
 Gradient descent is used all over the place in machine learning, not just for linear regression, but for training for example some of the most advanced neural network models.  gradient descent is an algorithm that you can use to try to minimize any function.<br>
 <img src="https://github.com/bilyatch/Machine_Learning/blob/main/photos/reduced%20funciton.png"><br>
 
 <h3>Gradient descent algorithm<h3><br>
  <img src="https://github.com/bilyatch/Machine_Learning/blob/main/photos/gradientdescent_algo.PNG"><br>
Here,Alpha is also called the learning rate. The learning rate is usually a small positive number between 0 and 1 and it might be say, 0.01. What Alpha does is, it basically controls how big of a step you take downhill. 

  <i><b><h3>Day four</h3></b></i>
<h3>Training Linear Regression</h3><br>
  <img src="https://user-images.githubusercontent.com/104844487/235959464-5ccf283f-4460-41a0-9465-85132cf8e9c3.png"><br>
here when derivative term is positive number then,
w=w-α.(positive no.),which decreases the value of w 
but, if the derivative term is negative then,
w=w-α.(negative no.),which increases the value of w.

<h3>Gradient descent for linear alogorithm</h3>
 <img src="https://user-images.githubusercontent.com/104844487/236116086-9357383c-b643-43f1-ad90-e30fa83e8d97.png"><br>
In Cost Function,depending on where you initialize the parameters w and b, you can end up at different local minima.
In Squared Error Cost Function,the cost function does not and will never have multiple local minima. It has a single global minimum because of this bowl-shape.It's a convex function. Which is as show in fig given below:<br>
  <img src="https://user-images.githubusercontent.com/104844487/236116595-1998f177-44ab-456a-a9c1-a5a9a45c6063.png">

  <i><b><h3>Day five</h3></b></i>
  <h4>Multi features</h4>
 Linear regression that look at not just one feature, but a lot of different features.<br>
   <img src="https://user-images.githubusercontent.com/104844487/236664860-f8156105-8143-489e-a7fc-c7fac9005428.png" >
  
  <h4>Vectorization</h4>
 When you're implementing a learning algorithm, using vectorization will both make your code shorter and also make it run much more efficiently. NumPy dot function is a vectorized implementation of the dot product operation between two vectors and especially when n is large, this will run much faster that's written as :<br>
  <b>f=np.dot(w,x)+b</b> <br>
  
  <h4>Gradient descent for multiple linear regression</h4>
 <img src="https://github.com/bilyatch/Machine_Learning/blob/6342c29e655a407438aab738e63f1f50381e276a/photos/gradientdescentwithmultiplelinearreagreassion.PNG" >
  

  <i><b><h3>Day six</h3></b></i>
  <h4>Feature scaling</h4>
  A technique called feature scaling that will enable gradient descent to run much faster.<br>
  If the given feature and parameters are:<br>
   <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/81778a38-8aeb-43fe-a05b-ababc0cd94f4"><br>
 Here, w1 tends to be multiplied by a very large number. In contrast, it takes a much larger change in w2 in order to change the predictions much. And thus small changes to w2, don't change the cost function nearly as much. So This is what might end up happening if you were to run great in discent, if you were to use your training data as is. Because the contours are so tall and skinny gradient descent may end up bouncing back and forth for a long time before it can finally find its way to the global minimum. Like given in fig. below<br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/775b6868-ab2f-43e5-b2cb-780b18558301"><br>
 So if we rescale x1 and x2 and now are both taking comparable ranges of values to each other. And if you run gradient descent on a cost function to find on this, rescaled x1 and x2 using this transformed data, then the contours will look more like this more like circles and less tall and skinny. And gradient descent can find a much more direct path to the global minimum as given below:<br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/5813fa89-9ea6-4aa1-9331-6714f6e104a0"><br>

  Lets consider:<br>
if x_1 ranges from 3 > x_1 > 2,000<br>
  One way to get a scale version of x_1 is to take each original <b> x1_ value and divide by 2,000 </b>, the maximum of the range i.e<br>
  <b>(3/2000) > x_1 > (2000/2000)</b><br>
  The scale x_1 will range from <b>0.15 > x_1 > 1</b>.
  <br>
  <br>
  <h4><b>Mean Normalization</b></h4>
  formula: <br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/a927e954-4ff4-42a7-acf3-b12631f0505a"><br>
  
  <h4><b>Z score normalization</b></h4>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuvlurKfnbEQWq0MujE2OwIgVTQzp-rCq9CQ&usqp=CAU"><br>
  <br>
  <br>
  acceptable scale ranges:<br>
  -1 ≤ x ≤ 1 <br>
  -3 ≤ x ≤ 3 <br>
  -0.3 ≤ x ≤ 0.3, It's okay, no rescaling is needed<br>
  0 ≤ x ≤ 3, It's also okay, no rescaling is needed <br>
  -2 ≤ x ≤ 0.5, It's also okay, no rescaling is needed <br>
  but,<br>
  -100 ≤ x ≤ 100 ,It's too large, so rescaling is needed <br>
  -0.001 ≤ x ≤ +0.001, It's too small, so rescaling is needed <br>
  76.23 ≤ x ≤ 230, It's too large, so rescaling is needed <br>
  
  
 <i><b><h3>Day seven</h3></b></i>
  <h4>Choosing the learning rate</h4>
   Choosing learning rates is an important part of training many learning algorithms. Your learning algorithm will run much better with an appropriate choice of learning rate. If it's too small, it will run very slowly and if it is too large, it may not even converge.<br>

   Sometimes you may see that the cost consistently increases after each iteration. This is also likely due to a learning rate that is too large, and it could be addressed by choosing a smaller learning rate also the cost can sometimes go up instead of decreasing and to fix this, you can use a smaller learning rate.<br>
   
   <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/5344dcc4-88a0-4102-ac64-992eade58cce"><br>
   
  <i><b><h3>Day eight</h3></b></i>
  <h4>Feature Engineering</h4>
  
Feature engineering, in which we might use out knowledge or intuition about the problem to design new features usually by transforming or combining the original features of the problem in order to make it easier for the learning algorithm to make accurate predictions. Depending on what insights we may have into the application, rather than just taking the features that we happen to have started off with sometimes by defining new features, we might be able to get a much better model. That's feature engineering.<br>
  
  <h4>Classification</h4>
  Classification where your output variable y can take on only one of a small handful of possible values instead of any number in an infinite range of numbers. It turns out that linear regression is not a good algorithm for classification problems. Classification problem where there are only two possible outputs is called binary classification.<br>
  
  <h4>Logistic Regression</h4>
 Logistic regression is a data analysis technique that uses mathematics to find the relationships between two data factors. It then uses this relationship to predict the value of one of those factors based on the other. The prediction usually has a finite number of outcomes, like yes or no. To build out to the logistic regression algorithm, there's an important mathematical function which is called the Sigmoid function, sometimes also referred to as the logistic function.<br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/8596fcff-bf1b-484b-95ef-cb299a0193dc"><br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/ca32bec2-5c71-40e7-aadd-3420834d0033"><br>


  
  
  
  
 
  


  



 
 

 
