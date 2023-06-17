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
  
  <i><b><h3>Day eigth</h3></b></i>
  <h4>Cost function for logistic regression</h4>
  We now know that the cost function gives you a way to measure how well a specific set of parameters fits the training data. Thereby gives you a way to try to choose better parameters. Recalling the linear regression cost function, cost function looks a convex function or a bowl shape or hammer shape. Gradient descent converge at the global minimum.<br>
  <br>
  Likewise if we try to use the same cost function as it was in linear regression then the cost function will not be a convex what's called a non-convex cost function and if we try to use gradient descent then we find a lots of local minima. So, For logistic regression Squared Error Cost is not a good choice. <br>
  Therefore, For the logistic regression the new cost function to make work on logistic regression will be:<br>
 <br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/e46f2206-13e8-4719-9abe-e16e91536721" style="float: center;" width="400" height="150"><br>
  
  <h3>Gradient Descent for logistic regression</h3>
  The Gradient descent for the logistic regression is different from logical regression since the f<sub>w,b</sub>(x) is different which is:<br>
  𝑓<sub>𝐰,𝑏</sub>(𝑥)=𝑔(𝑧)<br>
  where  𝑔(𝑧)<br>
  is the sigmoid function:<br>
  𝑔(𝑧)=(1/(1+𝑒<sup>−𝑧</sup>))<br>
 
  <h4>Problem of Overfitting and Underfitting</h4>
  The problem of <b>overfitting</b> in data analysis occurs when a predictive model becomes too complex or specific to the training data it was developed on, to the point where it fails to generalize well to new, unseen data. Overfitting is a common challenge in machine learning and statistical modeling, and it can lead to poor performance and inaccurate predictions. They couldn't end up with totally different predictions or highly variable predictions. That's why we say the algorithm has high variance.<br><br>
    The problem of <b>underfitting</b> in data analysis occurs when a predictive model is too simple or lacks complexity to capture the underlying patterns and relationships present in the data. Unlike overfitting, underfitting occurs when the model fails to adequately learn from the training data, leading to poor performance and low predictive power. There can be a clear pattern in the training data that the algorithm is just unable to capture. Another way to think of this form of bias is as if the learning algorithm has a very strong preconception, or we say a very strong bias<br><br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/416f019b-2a44-4f95-9162-40ba84b830a1" style="float: left;" width="400" height="200">
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/5317739a-722c-4520-b823-c7eabed5eaa0" style="float: right;" width="400" height="200"><br><br>
  
  <b>How to address overfitting</b><br>
  <ol>
  <li>Regularization</li>
  <li>Feature selection</li>
  <li>Increasing data size</li>
 <li> Cross-validation</li>
 <li> Early stopping</li>
  </ol>
  
  
  '<h4>Reguralization </h4>
  Regularization is a technique used to address the problem of overfitting in machine learning models. Overfitting occurs when a model learns the training data too well, to the point where it performs poorly on unseen data. Regularization helps to prevent overfitting by adding a penalty term to the model's loss function, discouraging complex or large parameter values. <br>
  The cost function of the regularized cost is given below:<br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/be020928-56df-48a1-b2d8-295241b6891e" style="float:right;" width="500" heigth="200"><br>
  
  <h4>Regularized in linear regression</h4>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/cd7e2d2d-56e1-4bd4-85ec-5b650ad980fe" width="800"
       heigth="400"><br>
<br>


  <h2>Neural Network</h2>
  <br>
  <h3><b><i>Demand prediction</i></b></h3>
  A neural network is a computational model inspired by the structure and function of the human brain. It consists of interconnected artificial neurons, organized into layers, that process and transmit information. Each neuron takes input, performs a mathematical operation on it, and produces an output that is then passed on to the next layer.<br>

we have a neural network with three layers: an input layer, a hidden layer, and an output layer.<br>
  <br>
<b>Input Layer:</b> The input layer consists of four neurons that receive the input data in the given picture below. Each neuron represents a feature or attribute of the input.<br>
  <br>
<b>Hidden Layer:</b> The hidden layer, in this case, consists of three neurons. These neurons perform calculations based on the input data and apply mathematical transformations to produce intermediate representations.<br>
  <br>
<b>Output Layer:</b> The output layer consists of one neurons that produce the final output of the neural network. These neurons combine the information from the hidden layer and generate the desired output.<br>
  
<img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/679f1d8f-a737-44f5-934b-bead0598649c"><br>
  <br>
  
  <h3>Recognizing Images</h3>
When it comes to image identification, neural networks excel at learning and recognizing patterns in visual data. Here's a general overview of how a neural network works for image identification:<br><br>

Data Preparation: The first step is to prepare the data for training the neural network. This involves collecting a labeled dataset of images, where each image is associated with a specific class or category (e.g., cat, dog, car).<br><br>

Network Architecture: Designing the neural network architecture is the next step. Typically, convolutional neural networks (CNNs) are used for image identification tasks. CNNs are specialized for processing grid-like data, such as images, by using convolutional layers to extract local features and pooling layers to downsample the data.<br><br>

Training: The neural network is trained using the labeled dataset. During training, the network adjusts its internal parameters (weights and biases) based on the input images and their corresponding labels. The process involves forward propagation, where the input image is passed through the network, and the output is compared to the expected output. The error between the predicted and expected output is then used to update the network's parameters through backpropagation.<br><br>

Feature Extraction: The convolutional layers in the network extract various features from the input image at different levels of abstraction. The initial layers capture low-level features like edges, textures, and colors, while deeper layers learn more complex features like shapes and object parts.<br><br>

Classification: The extracted features are fed into fully connected layers, which learn to classify the image based on the extracted information. The output layer consists of neurons, each representing a specific class or category. The neuron with the highest activation indicates the predicted class of the input image.<br><br>

Prediction: Once the neural network is trained, it can be used for image identification on unseen images. The input image is passed through the network, and the network produces a probability distribution over the possible classes. The class with the highest probability is considered the predicted label for the image.<br><br>

Fine-tuning and Optimization: Fine-tuning techniques, such as regularization and optimization algorithms, can be applied to improve the network's performance and generalize well to new, unseen images.<br><br>

By repeatedly adjusting the network's parameters based on the training data, the neural network learns to recognize visual patterns and becomes capable of accurately identifying images from different classes.<br>

 In the given fig. below it has given the steps for detecting the face:<br>
<img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/fa1ae1b1-4ba9-4c63-9c09-d76f332c992e">
  <br>
  
  
  <h3>Notation of complex neural network</h3>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/896901d3-6a34-43a2-b3a7-285e7045cab6", height="400",width="600">
  <br>
  <br>
  <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/417fc75c-0fa4-48e2-817b-8dd1c369d8d3"><br><br>
  
  <h3>Specualation of Artificial General Intelligence<(AGI)</h3>
   AI actually includes two very different things. One is ANI which stands for artificial narrow intelligence. This is an AI system that does one thing, a narrow task, sometimes really well and can be incredibly valuable, such as the smart speaker or self-driving car or web search, or AI applied to specific applications such as farming or factories. ANI is a subset of AI, the rapid progress in ANI makes it logically true that AI has also made tremendous progress in the last decade. There's a different idea in AI, which is AGI, artificial general intelligence. There's hope of building AI systems that could do anything a typical human can do.<br>
   <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/2f5f9537-a689-4f5a-8eb5-47f179e39287">

   <h3>Model training steps</h3>
   <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/ffb71b7d-6aae-4e68-b781-e28d25a2ffee">
   

   <h3>example of activation function</h3>
   <img src="https://github.com/bilyatch/Machine_Learning/assets/104844487/156b7c0b-3f46-4b9c-a079-d7e532d2151c">
   



  
  
  
  
 
  


  



 
 

 
