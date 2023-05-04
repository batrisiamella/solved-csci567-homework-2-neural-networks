Download Link: https://assignmentchef.com/product/solved-csci567-homework-2-neural-networks
<br>
Figure 1: A diagram of a 1 hidden-layer multi-layer perceptron (MLP). <em>The edges mean mathematical operations, and the circles mean variables. </em>Generally we call the combination of a linear (or affine) operation and a nonlinear operation (like element-wise sigmoid or the rectified linear unit (relu) operation as in eq. (3)) as a hidden layer.

<strong>Algorithmic component</strong>

Neural networks (error-backpropagation, initialization, and nonlinearity)

[Recommended maximum time spent: 1 hour]

In the lecture (see lec8.pdf), we have talked about error-backpropagation, a way to compute partial derivatives (or gradients) w.r.t the parameters of a neural network. We have also mentioned that optimization is challenging and nonlinearity is important for neural networks. In this question, you are going to (Q1.1) practice error-backpropagation, (Q1.2) investigate how initialization affects optimization, and (Q1.3) the importance of nonlinearity.

Specifically, you are given the following 1-hidden layer multi-layer perceptron (MLP) for a <em>K</em>class classification problem (see Fig. 1 for illustration and details), and (<em>x </em>∈ R<em><sup>D</sup>,y </em>∈ {1<em>,</em>2<em>,</em>··· <em>,K</em>})

<table width="0">

 <tbody>

  <tr>

   <td width="285">is a labeled instance,</td>

   <td width="319"> </td>

   <td width="20"> </td>

  </tr>

  <tr>

   <td width="285"><em>x </em>∈ R<em><sup>D</sup></em></td>

   <td width="319"> </td>

   <td width="20">(1)</td>

  </tr>

  <tr>

   <td width="285"><em>u </em>= <em>W</em>(1)<em>x </em>+ <em>b</em>(1)</td>

   <td width="319"><em>,</em><em>W</em>(1) <sup>∈ </sup>R<em>M</em>×<em>D </em>and <em>b</em>(1) <sup>∈ </sup>R<em>M</em></td>

   <td width="20">(2)</td>

  </tr>

 </tbody>

</table>

<em>h</em>                                                                                     (3)

<table width="0">

 <tbody>

  <tr>

   <td width="123"><em>a </em>= <em>W</em>(2)<em>h </em>+ <em>b</em>(2)</td>

   <td width="338"><em>,</em><em>W</em>(2) <sup>∈ </sup>R<em>K</em>×<em>M </em>and <em>b</em>(2) <sup>∈ </sup>R<em>K</em></td>

   <td width="20">(4)</td>

  </tr>

 </tbody>

</table>

<em>z</em>                                                                                                                            (5)

<em>y</em>ˆ = argmax<em><sub>k </sub>z<sub>k</sub>.                                                                                                                         </em>(6)

For <em>K</em>-class classification problem, one popular loss function for training is the cross-entropy loss,

<em>l </em>= −<sup>X</sup><strong>1</strong>[<em>y </em>== <em>k</em>]log<em>z<sub>k</sub>,                                                                                    </em>(7)

<em>k</em>

where <strong>1</strong>[True] = 1; otherwise, 0<em>.                                                                </em>(8)

For ease of notation, let us define the one-hot (i.e., 1-of-<em>K</em>) encoding

<em>y </em>∈ R<em><sup>K </sup></em>and(9)

<em>.</em>

so that

<em>.                                             </em>(10)

<em>∂l</em>

<strong>Q1.1            </strong>Assume that you have computed <em>u</em>, <em>h</em>, <em>a</em>, <em>z</em>, given (<em>x</em>, <em>y</em>). Please first express          in terms

<em>∂</em><em>u</em>

<h2><em>∂l</em></h2>

of            , <em>u</em>, <em>h</em>, and <em>W</em><sup>(2)</sup>.

<h2><em>∂</em><em>a</em></h2>

<em>∂l</em>

=? <em>∂</em><em>u</em>

<h3>∂l</h3>

Then express      in terms of <em>z </em>and <em>y</em>.

<h2><em>∂</em><em>a</em></h2>

<em>∂l</em>

=?

<em>∂</em><em>a</em>

<h3>                                           ∂l                    ∂l                               ∂l                                             ∂l                                 ∂l</h3>

Finally, compute                   and              in terms of     and <em>x</em>. Compute     in terms of    and <em>h</em>.

<em>∂</em><em>W</em>(1)                    <em>∂</em><em>b</em>(1)                                         <em>∂</em><em>u                                         ∂</em><em>W</em>(2)                                          <em>∂</em><em>a</em>

=?

=?

=?

You only need to write down the final answers of the above 5 question marks. You are encouraged to use matrix/vector forms to simplify your answers. Note that max{0<em>,u</em>} is not differentiable w.r.t. <em>u </em>at <em>u </em>= 0. Please note that

<em>,</em>

<table width="0">

 <tbody>

  <tr>

   <td width="466">which stands for the Heaviside step function. You can use<em>∂ </em>max{0<em>,</em><em>u</em>}</td>

   <td width="158"> </td>

  </tr>

  <tr>

   <td width="466">= <em>H</em>(<em>u</em>)</td>

   <td width="158">(12)</td>

  </tr>

 </tbody>

</table>

(11)

<em>∂</em><em>u</em>

<h3>∂l</h3>

in your derivation of         .

<h2><em>∂</em><em>u</em></h2>

You can also use ·∗ to represent element-wise product between two vectors or matrices. For example,

<em>v</em>, where <em>v </em>∈ R<em><sup>I </sup></em>and <em>c </em>∈ R<em><sup>I</sup>.                                               </em>(13)

Also note that the partial derivatives of the loss function w.r.t. the variables (e.g., a scalar, a vector, or a matrix) will have the same shape as the variables.

<em>What to submit: </em>No more than 5 lines of derivation for each of the 5 partial derivatives.

<strong>Q1.2 </strong>Suppose we initialize <em>W</em><sup>(1)</sup>, <em>W</em><sup>(2)</sup>, <em>b</em><sup>(1) </sup>with zero matrices/vectors (i.e., matrices and vectors with all elements set to 0), please first verify that are all zero matrices/vectors,

irrespective of <em>x</em>, <em>y </em>and the initialization of <em>b</em><sup>(2)</sup>.

Now if we perform stochastic gradient descent for learning the neural network using a training set, please explain with a concise mathematical statement in one sentence why no learning will happen on <em>W</em><sup>(1)</sup>, <em>W</em><sup>(2)</sup>, <em>b</em><sup>(1) </sup>(i.e., they will not change no matter how many iterations are run). Note that this will still be the case even with weight decay and momentum if the initial velocity vectors/matrices are set to zero.

<em>What to submit: </em>No submission for the verification question. Your concise mathematical statement in one sentence for the explanation question.

<strong>Q1.3 </strong>As mentioned in the lecture (see lec8.pdf), nonlinearity is very important for neural networks. With nonlinearity (e.g., eq. (3)), the neural network shown in Fig. 1 can bee seen as a nonlinear basis function <em>φ </em>(i.e., <em>φ</em>(<em>x</em>) = <em>h</em>) followed by a linear classifier <em>f </em>(i.e., <em>f</em>(<em>h</em>) = <em>y</em>ˆ).

Please show that, by removing the nonlinear operation in eq. (3) and setting eq. (4) to be <em>a </em>= <em>W</em><sup>(2)</sup><em>u </em>+ <em>b</em><sup>(2)</sup>, the resulting network is essentially a linear classifier. More specifically, you can now represent <em>a </em>as <em>Ux </em>+ <em>v</em>, where <em>U </em>∈ R<em><sup>K</sup></em><sup>×<em>D </em></sup>and <em>v </em>∈ R<em><sup>K</sup></em>. Please write down the representation of <em>U </em>and <em>v </em>using <em>W</em>(1)<em>,</em><em>W</em>(2)<em>,</em><em>b</em>(1)<em>, </em>and <em>b</em>(2)

<em>U </em>=? <em>v </em>=?

<em>What to submit: </em>No more than 2 lines of derivation for each of the question mark.

<h1>2           Kernel methods</h1>

[Recommended maximum time spent: 1 hour]

In the lecture (see lec10.pdf) , we have seen the “kernelization” of regularized least squares problem. The “kernelization” process depends on an important observation: the optimal model parameter can be expressed as a linear combination of the transformed features. You are now to prove a more general case.

Consider a convex loss function in the form <em>`</em>(<em>w</em><sup>T</sup><em>φ</em>(<em>x</em>)<em>,y</em>)<em>, </em>where <em>φ</em><strong>(</strong><em>x</em><strong>) </strong>∈ R<em><sup>M </sup></em>is a nonlinear feature mapping, and <em>y </em>is a label or a continuous response value.

Now solve the regularized loss minimization problem on a training set D = {(<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>N</sub>,y<sub>N</sub></em>)},

(14)

<strong>Q2.1 </strong>Show that the optimal solution of <em>w </em>can be represented as a linear combination of the training samples. You can assume <em>`</em>(<em>s,y</em>) is differentiable w.r.t. <em>s</em>, i.e. during derivation, you can use the derivative and assume it is a known quantity.

<em>What to submit: </em>Your fewer than 10 line derivation and optimal solution of <em>w</em>.

<strong>Q2.2 </strong>Assume the combination coefficient is <em>α<sub>n </sub></em>for <em>n </em>= 1<em>,…,N</em>. Rewrite loss function Eqn. 14 in terms of <em>α<sub>n </sub></em>and kernel function value <em>K<sub>ij </sub></em>= <em>k</em>(<em>x<sub>i</sub>,</em><em>x<sub>j</sub></em>).

<em>What to submit: </em>Your objective function in terms of <em>α </em>and <em>K</em>.

<strong>Q2.3 </strong>After you obtain the general formulation for Q2.1 and Q2.2, please plug in three different loss functions we have seen so far, and examine what you get.

square loss:

<em>                                                                                  y </em>∈ R                      (15)

cross entropy loss:

(16)

<table width="0">

 <tbody>

  <tr>

   <td width="470">perceptron loss:</td>

   <td width="99"> </td>

   <td width="28"> </td>

  </tr>

  <tr>

   <td width="470"><em>`</em>(<em>w</em><sup>T</sup><em>φ</em>(<em>x</em>)<em>,y</em>) = max(−<em>y</em><em>w</em><sup>T</sup><em>φ</em>(<em>x</em>)<em>,</em>0)</td>

   <td width="99"><em>y </em>∈ {−1<em>,</em>1}</td>

   <td width="28">(17)</td>

  </tr>

 </tbody>

</table>

<em>What to submit: </em>Nothing.

<strong>Programming component</strong>

<h1>3           High-level descriptions</h1>

<h2>3.1         Dataset</h2>

We will use <strong>mnist subset </strong>(images of handwritten digits from 0 to 9). This is the same subset of the full MNIST that we used for Homework 1. As before, the dataset is stored in a JSON-formated file <strong>mnist subset.json</strong>. You can access its training, validation, and test splits using the keys ‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load <strong>mnist subset.json </strong>to the variable <em>x</em>. Then, <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>] refers to the training set of <strong>mnist subset</strong>. This set is a list with two elements: <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>][0] containing the features of size <em>N </em>(samples) ×<em>D </em>(dimension of features), and <em>x</em>[<sup>0</sup><em>train</em><sup>0</sup>][1] containing the corresponding labels of size <em>N</em>.

<h2>3.2         Tasks</h2>

You will be asked to implement 10-way classification using multinomial logistic regression (Sect. 4) and neural networks (Sect. 5). Specifically, you will

<ul>

 <li>finish the implementation of all python functions in our template code.</li>

 <li>run your code by calling the specified scripts to generate output files.</li>

 <li>add, commit, and push (1) all *.py files, and (2) all *.json files that you have created.</li>

</ul>

In the next two subsections, we will provide a <strong>high-level </strong>checklist of what you need to do. Furthermore, as in the previous homework, you are not responsible for loading/pre-processing data; we have done that for you. For specific instructions, please refer to text in Sect. 4 and Sect. 5, as well as corresponding python scripts.

<h3><strong>3.2.1        </strong><strong>Multi-class Classification</strong></h3>

<strong>Coding </strong>In logistic prog.py, finish implementing the following functions: logistic train ovr, logistic test ovr, logistic mul train, and logistic mul test. Refer to logistic prog.py and Sect. 4 for more information.

<strong>Running your code </strong>Run the script q43.sh after you finish your implementation. This will output logistic res.json.

<strong>What to submit </strong>Submit both logistic prog.py and logistic res.json.

<strong>3.2.2      Neural networks</strong>

<strong>Preparation </strong>Read dnn mlp.py and dnn cnn.py.

<h3><strong>Coding</strong></h3>

First, in dnn misc.py, finish implementing

<ul>

 <li>forward and backward functions in class linear layer</li>

 <li>forward and backward functions in class relu</li>

 <li>backward function in class dropout (before that, please read forward function).</li>

</ul>

Refer to dnn misc.py and Sect. 5 for more information.

Second, in dnn cnn 2.py, finish implementing the main function. There are three TODO items. Refer to dnn cnn 2.py and Sect. 5 for more information.

<strong>Running your code </strong>Run the scripts q53.sh, q54.sh, q55.sh, q56.sh, q57.sh, q58.sh, q510.sh after you finish your implementation. This will generate, respectively,

MLP lr0.01 m0.0 w0.0 d0.0.json

MLP lr0.01 m0.0 w0.0 d0.5.json

MLP lr0.01 m0.0 w0.0 d0.95.json

LR lr0.01 m0.0 w0.0 d0.0.json CNN lr0.01 m0.0 w0.0 d0.5.json

CNN lr0.01 m0.9 w0.0 d0.5.json

CNN2 lr0.001 m0.9 w0.0 d0.5.json

<strong>What to submit </strong>Submit dnn misc.py, dnn cnn 2.py, and seven .json files that are generated from the scripts.

<h2>3.3         Cautions</h2>

Please do not import packages that are not listed in the provided code. Follow the instructions in each section strictly to code up your solutions. <strong>Do not change the output format</strong>. <strong>Do not modify the code unless we instruct you to do so</strong>. A homework solution that does not match the provided setup, such as format, name, initializations, etc., <strong>will not </strong>be graded. It is your responsibility to <strong>make sure that your code runs with the provided commands and scripts on the VM</strong>. Finally, make sure that you <strong>git add, commit, and push all the required files</strong>, including your code and generated output files.

<h2>3.4         Advice</h2>

We are extensively using softmax and sigmoid function in this homework. To avoid numerical issues such as overflow and underflow caused by <em>numpy.exp</em>() and <em>numpy.log</em>(), please use the following implementations:

<ul>

 <li>Let <em>x </em>be a input vector to the softmax function. Use <em>x</em>˜ = <em>x </em>− max(<em>x</em>) instead of using <em>x </em>directly for the softmax function <em>f</em>, i.e.</li>

 <li>If you are using <em>log</em>(), make sure the input to the log function is positive. Also, there may be chances that one of the outputs of softmax, e.g. <em>f</em>(<em>x</em>˜<em><sub>i</sub></em>), is extremely small but you need the value ln(<em>f</em>(<em>x</em>˜<em><sub>i</sub></em>)), you can convert the computation into <em>x</em>˜</li>

</ul>

We have implemented and run the code ourselves without any problems, so if you follow the instructions and settings provided in the python files, you should not encounter overflow or underflow.

<h1>4           Multi-class Classification</h1>

You will modify 4 python functions in logistic prog.py. First, you will implement two functions that train and test a one-versus-rest multi-class classification model. Second, you will implement two functions that train and test a multinomial logistic regression model. Finally, you will run the command that train and test the two models using your implemented functions, and our code will automatically store your results to logistic res.json.

<h2>Coding: One-versus-rest</h2>

<strong>Q4.1 </strong>Implement the code to solve the multi-class classification task with the one-versus-rest strategy. That is, train 10 binary logistic regression models following the setting provided in class: for each class <em>C<sub>k</sub>,k </em>= 1<em>,</em>··· <em>,</em>10, we create a binary classification problem as follows:

<ul>

 <li>Re-label training samples with label <em>C<sub>k </sub></em>as positive (namely 1)</li>

 <li>Re-label other samples as negative (namely 0)</li>

</ul>

We wrote functions to load, relabel, and sample the data for you, so you are not responsible for doing it.

<strong>Training </strong>Finish the implementation of the function logistic train ovr(Xtrain, ytrain, w, b, step size, max iterations). As in the previous homework, we have pre-defined the hyperparameters and initializations in the template code. Moreover, you will use the <strong>AVERAGE </strong>of gradients from all training samples to update the parameters.

<strong>Testing </strong>Finish the implementation of the function logistic test ovr(Xtest, w l, b l). This function should return the predicted probability, i.e., the value output by logistic function without thresholding, instead of the 0/1 label. Formally, for each test data point <em>x<sub>i</sub></em>, we get its final prediction by ˆ<em>y<sub>i </sub></em>= argmax<em><sub>k</sub></em><sub>∈{1<em>,</em>···<em>,</em>10} </sub><em>f<sub>k</sub></em>(<em>x<sub>i</sub></em>), where ˆ<em>y<sub>i </sub></em>is the predicted label and <em>f<sub>k</sub></em>(<em>x<sub>i</sub></em>) is the predicted probability by the <em>k<sup>th </sup></em>logistic regression model <em>f<sub>k</sub></em>. Then, you compute the classification accuracy as follows:

<em>,                                                                   </em>(18)

where <em>y<sub>i </sub></em>is the ground-truth label of <em>x<sub>i </sub></em>and <em>N<sub>test </sub></em>is the total number of test data instances.

<em>What to do and submit: </em>Your logistic prog.py with completed logistic train ovr and logistic test ovr.

<h2>Coding: Multinomial logistic regression</h2>

<strong>Q4.2 </strong>Implement the multinomial logistic regression, training a 10-way classifier (with the softmax function) on <strong>mnist subset </strong>dataset.

<strong>Training </strong>Finish the implementation of the function logistic mul train(Xtrain, ytrain, w, b, step size, max iterations). Again, we have pre-defined the hyper-parameters and initializations in the template code. Moreover, you will use the <strong>AVERAGE </strong>of gradients from all training samples to update the parameters.

<strong>Testing </strong>Finish the implementation of the function logistic mul test(Xtest, w l, b l) For each test data point <em>x<sub>i</sub></em>, compute ˆ<em>y </em>= argmax<em><sub>k</sub></em><sub>∈{1<em>,</em>···<em>,</em>10} </sub><em>p</em>(<em>y </em>= <em>k</em>|<em>x</em>), where <em>p</em>(<em>y </em>= <em>k</em>|<em>x</em>) is the predicted probability by the multinomial logistic regression. Then, compute the accuracy following Eqn. 18.

<em>What to do and submit: </em>Your logistic prog.py with completed logistic mul train and logistic mul test.

<h2>Training and getting generated output files from both one-versus-rest and multinomial logistic regression models</h2>

<strong>Q4.3 </strong><em>What to do and submit: </em>run script q43.sh. It will generate logistic res.json. Add, commit, and push both logistic prog.py and logistic res.json before the due date. <em>What it does: </em>q43.sh will run python3 logistic prog.py. This will train your models (for both Q4.1 and Q4.2 above) and test the trained models (for both Q4.1 and Q4.2 above). The output file stores accuracies of both models.

<h1>5           Neural networks: multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs)</h1>

In recent years, neural networks have been one of the most powerful machine learning models. Many toolboxes/platforms (e.g., TensorFlow, PyTorch, Torch, Theano, MXNet, Caffe, CNTK) are publicly available for efficiently constructing and training neural networks. The core idea of these toolboxes is to treat a neural network as a combination of <em>data transformation modules</em>. For example, in Fig. 2, the edges correspond to module names of the same neural network shown in Fig. 1 and Sect. 1.

Now we will provide more information on modules for this homework. Each module has its own parameters (but note that a module may have no parameters). Moreover, each module can perform a forward pass and a backward pass. The forward pass performs the computation of the module, given the input to the module. The backward pass computes the partial derivatives of the loss function w.r.t. the input and parameters, given the partial derivatives of the loss function w.r.t. the output of the module. Consider a module hmodule namei. Let hmodule namei.forward and hmodule namei.backward be its forward and backward passes, respectively.

input features                                                                                                                                                           predicted label

Figure 2: A diagram of a 1-hidden layer multi-layer perceptron (MLP), with modules indicated on the edges. The circles correspond to variables. The <em>rectangles </em>shown in Fig. 1 are removed for clearness. The term <em>relu </em>stands for rectified linear units.

For example, the linear module may be defined as follows.

forward pass:              <em>u </em>= linear<sup>(1)</sup>.forward(<em>x</em>) = <em>W</em><sup>(1)</sup><em>x </em>+ <em>b</em><sup>(1)</sup><em>,                                           </em>(19)

where <em>W</em><sup>(1) </sup>and <em>b</em><sup>(1) </sup>are its parameters.

backward pass:           [] = linear<sup>(1)</sup>.backward(<em>x,   </em><em>.                       </em>(20)

<em>∂</em><em>x ∂</em><em>W</em>(1) <em>∂</em><em>b</em>(1)                                                                                         <em>∂</em><em>u</em>

Let us assume that we have implemented all the desired modules. Then, getting <em>y</em>ˆ for <em>x </em>is equivalent to running the forward pass of each module in order, given <em>x</em>. All the intermediated variables (i.e., <em>u</em>, <em>h</em>, etc.) will all be computed along the forward pass. Similarly, getting the partial derivatives of the loss function w.r.t. the parameters is equivalent to running the backward pass of

<h2><em>∂l</em></h2>

each module in a reverse order, given          .

<h2><em>∂</em><em>z</em></h2>

In this question, we provide a Python environment based on the idea of modules. Every module is defined as a class, so you can create multiple modules of the same functionality by creating multiple object instances of the same class. Your work is to finish the implementation of several modules, where these modules are elements of a multi-layer perceptron (MLP) or a convolutional neural network (CNN). We will apply these models to the same 10-class classification problem introduced in Sect. 4. We will train the models using stochastic gradient descent with minibatch, and explore how different hyperparameters of optimizers and regularization techniques affect training and validation accuracies over training epochs. For deeper understanding, check out, e.g., the seminal work of Yann LeCun et al. “Gradient-based learning applied to document recognition,” written in 1998.

We give a specific example below. Suppose that, at iteration <em>t</em>, you sample a mini-batch of

<em>N </em>examples  from the training set (<em>K </em>= 10). Then, the loss of such a mini-batch given by Fig. 2 is (softmax.forward(linear<sup>(2)</sup>.forward(relu.forward(linear<sup>(1)</sup>.forward(<em>x<sub>i</sub></em>))))<em>,</em><em>y<sub>i</sub></em>)                (21)

(softmax.forward(linear<sup>(2)</sup>.forward(relu.forward(<em>u<sub>i</sub></em>)))<em>,</em><em>y<sub>i</sub></em>)                                            (22)

= ···                                                                                                                                                                                (23)

<em>N</em>

(softmax.forward(<em>a<sub>i</sub></em>)<em>,</em><em>y<sub>i</sub></em>)                                                                                                                (24)

<em>.                                                                                                                                         </em>(25)

That is, in the forward pass, we can perform the computation of a certain module to all the <em>N </em>input examples, and then pass the <em>N </em>output examples to the next module. This is the same case for the backward pass. For example, according to Fig. 2, if we are now to pass the partial derivatives of the loss w.r.t.  to linear<sup>(2)</sup>.backward, then

<em> .                                                                    </em>(26)

linear<sup>(2)</sup>.backward will then compute and pass it back to relu.backward.

<h2>Preparation</h2>

<strong>Q5.1 </strong>Please read through dnn mlp.py and dnn cnn.py. Both files will use modules defined in dnn misc.py (which you will modify). Your work is to understand how modules are created, how they are linked to perform the forward and backward passes, and how parameters are updated based on gradients (and momentum). The architectures of the MLP and CNN defined in dnn mlp.py and dnn cnn.py are shown in Fig. 3 and Fig. 4, respectively.

<em>What to submit: </em>Nothing.

<h2>Coding: Modules</h2>

(1)             dropout (2)

Figure 3: The diagram of the MLP implemented in dnn mlp.py. The circles mean variables and edges mean modules.

max pooling                                                                                                             dropout

Figure 4: The diagram of the CNN implemented in dnn cnn.py. The circles correspond to variables and edges correspond to modules. Note that the input to CNN may not be a vector (e.g., in dnn cnn.py it is an image, which can be represented as a 3-dimensional tensor). The flatten layer is to reshape its input into vector.

<strong>Q5.2 </strong>You will modify dnn misc.py. This script defines all modules that you will need to construct the MLP and CNN in dnn mlp.py and dnn cnn.py, respectively. You have three tasks. First, finish the implementation of forward and backward functions in class linear layer. Please follow Eqn. (2) for the forward pass. Second, finish the implementation of forward and backward functions in class relu. Please follow Eqn. (3) for the forward pass and Eqn. (11) for deriving the partial derivatives (note that relu itself has no parameters). Third, finish the the implementation of backward function in class dropout. We define the forward pass and the backward pass as follows.

forward pass:           <em>s </em>= dropout.forward(<em> ,                    </em>(27)

where <em>p<sub>j </sub></em>is sampled uniformly from [0<em>,</em>1)<em>,</em>∀<em>j </em>∈ {1<em>,</em>··· <em>,J</em>}<em>, </em>and <em>r </em>∈ [0<em>,</em>1) is a pre-defined scalar named dropout rate<em>.</em>

<h3>∂l</h3>

backward pass:           = dropout.backward(                                                                                   <em>.          </em>(28)

<h3>∂q</h3>

Note that <em>p<sub>j</sub>,j </em>∈ {1<em>,</em>··· <em>,J</em>} and <em>r </em>are not be learned so we do not need to compute the derivatives w.r.t. to them. Moreover, <em>p<sub>j</sub>,j </em>∈ {1<em>,</em>··· <em>,J</em>} are re-sampled every forward pass, and are kept for the following backward pass. The dropout rate <em>r </em>is set to 0 during testing.

Detailed descriptions/instructions about each pass (i.e., what to compute and what to return) are included in dnn misc.py. Please do read carefully.

Note that in this script we do import numpy as np. Thus, to call a function XX from numpy, please u np.XX.

<em>What to do and submit: </em>Finish the implementation of 5 functions specified above in dnn misc.py. Submit your completed dnn misc.py.

<h2>Testing dnn misc.py</h2>

<strong>Q5.3              </strong><em>What to do and submit: </em>run script q53.sh. It will output MLP lr0.01 m0.0 w0.0 d0.0.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q53.sh will run python3 dnn mlp.py with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.0. The output file stores the training and validation accuracies over 30 training epochs.

<strong>Q5.4              </strong><em>What to do and submit: </em>run script q54.sh. It will output MLP lr0.01 m0.0 w0.0 d0.5.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q54.sh will run python3 dnn mlp.py –dropout rate 0.5 with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.

<strong>Q5.5              </strong><em>What to do and submit: </em>run script q55.sh. It will output MLP lr0.01 m0.0 w0.0 d0.95.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q55.sh will run python3 dnn mlp.py –dropout rate 0.95 with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.95. The output file stores the training and validation accuracies over 30 training epochs.

You will observe that the model in Q5.4 will give better validation accuracy (at epoch 30) compared to Q5.3. Specifically, dropout is widely-used to prevent over-fitting. However, if we use a too large dropout rate (like the one in Q5.5), the validation accuracy (together with the training accuracy) will be relatively lower, essentially under-fitting the training data.

<strong>Q5.6 </strong><em>What to do and submit: </em>run script q56.sh. It will output LR lr0.01 m0.0 w0.0 d0.0.json. Add, commit, and push this file before the due date.

<em>What it does: </em>q56.sh will run python3 dnn mlp nononlinear.py with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.0. The output file stores the training and validation accuracies over 30 training epochs.

The network has the same structure as the one in Q5.3, except that we remove the relu (nonlinear) layer. You will see that the validation accuracies drop significantly (the gap is around 0.03). Essentially, without the nonlinear layer, the model is learning multinomial logistic regression similar to Q4.2.

<strong>Q5.7              </strong><em>What to do and submit: </em>run script q57.sh. It will output CNN lr0.01 m0.0 w0.0 d0.5.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q57.sh will run python3 dnn cnn.py with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.

max-p        max-p                                                                                                dropout

Figure 5: The diagram of the CNN you are going to implement in dnn cnn 2.py. The term <em>conv </em>stands for convolution; <em>max-p </em>stands for max pooling. The circles correspond to variables and edges correspond to modules. Note that the input to CNN may not be a vector (e.g., in dnn cnn 2.py it is an image, which can be represented as a 3-dimensional tensor). The flatten layer is to reshape its input into vector.

<strong>Q5.8              </strong><em>What to do and submit: </em>run script q58.sh. It will output CNN lr0.01 m0.9 w0.0 d0.5.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q58.sh will run python3 dnn cnn.py –alpha 0.9 with learning rate 0.01, momentum 0.9, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.

You will see that Q5.8 will lead to faster convergence than Q5.7 (i.e., the training/validation accuracies will be higher than 0.94 after 1 epoch). That is, using momentum will lead to more stable updates of the parameters.

<h2>Coding: Building a deeper architecture</h2>

<strong>Q5.9 </strong>The CNN architecture in dnn cnn.py has only one convolutional layer. In this question, you are going to construct a two-convolutional-layer CNN (see Fig. 5 using the modules you implemented in Q5.2. Please modify the main function in dnn cnn 2.py. The code in dnn cnn 2.py is similar to that in dnn cnn.py, except that there are a few parts marked as TODO. You need to fill in your code so as to construct the CNN in Fig. 5.

<em>What to do and submit: </em>Finish the implementation of the main function in dnn cnn 2.py (search for TODO in main). Submit your completed dnn cnn 2.py.

<h2>Testing dnn cnn 2.py</h2>

<strong>Q5.10                      </strong><em>What to do and submit: </em>run script q510.sh. It will output CNN2 lr0.001 m0.9 w0.0 d0.5.json.

Add, commit, and push this file before the due date.

<em>What it does: </em>q510.sh will run python3 dnn cnn 2.py –alpha 0.9 with learning rate 0.01, momentum 0.9, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.

You will see that you can achieve slightly higher validation accuracies than