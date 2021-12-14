# SVM_from_scratch

![Untitled](SVM_from_scratch%20c7258e6a31c84f9a883814224f083139/Untitled.png)

In this project, we have implemented a support vector classier to solve a binary class classication
task. The task is distinguishing between images of handwritten digits from the MNIST database.
Each sample is a 28x28 gray scale image, associated with a label. The project contains 3 different implementations grouped as Q1, Q2 and Q2 as show in the repository structure

### Repository structure

```

**SVM_from_scratch**
│   README.md
│   file001.txt
|   data.py
|   train-images-idx3-ubyte
|   train-labels-idx1-ubyte
└───Q1
│   │   SVM.py
│   │   run_1.py
|
└───Q2
│   │   SVM_decompose.py
│   │   run_2.py
|
└───Q3
│   │   MVP.py
│   │   run_3.py

```

### Q1

In this implementation , we will only consider numbers the images and labels of numbers 3 and 8. To solve this task, we used "SLSQP" constraint optimizer from scipy library which solves the dual soft margin svm quadratic optimization problem with a linear/rbf/polynomial kernel. We have splitted data into train/test with fixed seed and 80%/20% proportions.

### Q2

Here we have implemented the SVM light decomposition method. SVM light is an iterative method where in each step a sub set of data points are selected (which is called the working set) and SLSQP algorithm solves the quadratic optimization problem for the subset. Here the crucial part is selection of the subset at each iteration and the stopping criteria of the decomposition algorithm. The SVM light algorithm suggests choosing  q/2 indices from <!-- $R(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=R(%5Calpha%5Ek)"> and q/2 indices from  <!-- $S(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=S(%5Calpha%5Ek)">  where q cardinality of the working set and <!-- $R(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=R(%5Calpha%5Ek)"> and <!-- $S(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=S(%5Calpha%5Ek)"> are shown in formulas bellow.

<!-- $S(\alpha^k) = \{i:(\alpha_i \leq C\ \&\ y_i = -1), (\alpha_i \geq 0\ \&\ y_i = 1)\}\\$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=S(%5Calpha%5Ek)%20%3D%20%5C%7Bi%3A(%5Calpha_i%20%5Cleq%20C%5C%20%5C%26%5C%20y_i%20%3D%20-1)%2C%20(%5Calpha_i%20%5Cgeq%200%5C%20%5C%26%5C%20y_i%20%3D%201)%5C%7D%5C%5C">

<!-- $R(\alpha^k) = \{i:(\alpha_i \leq C\ \&\ y_i = 1), (\alpha_i \geq 0\ \&\ y_i = -1)\}\\$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=R(%5Calpha%5Ek)%20%3D%20%5C%7Bi%3A(%5Calpha_i%20%5Cleq%20C%5C%20%5C%26%5C%20y_i%20%3D%201)%2C%20(%5Calpha_i%20%5Cgeq%200%5C%20%5C%26%5C%20y_i%20%3D%20-1)%5C%7D%5C%5C">

. Based on that, the selection rule is defined as,

- select first <!-- $q_1 = q/2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=q_1%20%3D%20q%2F2"> indices from <!-- $R(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=R(%5Calpha%5Ek)"> such that <!-- $R(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=R(%5Calpha%5Ek)"> is sorted as bellow;
<!-- $\frac{- \nabla f(\alpha^k){i^1(k)}}{y{i^1(k)}} \geq \frac{- \nabla f(\alpha^k){i^2(k)}}{y{i^2(k)}} \geq ... \geq \frac{- \nabla f(\alpha^k){i^{q_1}(k)}}{y{i^{q_1}(k)}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bi%5E1(k)%7D%7D%7By%7Bi%5E1(k)%7D%7D%20%5Cgeq%20%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bi%5E2(k)%7D%7D%7By%7Bi%5E2(k)%7D%7D%20%5Cgeq%20...%20%5Cgeq%20%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bi%5E%7Bq_1%7D(k)%7D%7D%7By%7Bi%5E%7Bq_1%7D(k)%7D%7D">
- select first $q_2 = q/2$ indices from <!-- $S(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=S(%5Calpha%5Ek)"> such that <!-- $S(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=S(%5Calpha%5Ek)"> is sorted as bellow;
<!-- $\frac{- \nabla f(\alpha^k){j^1(k)}}{y{j^1(k)}} \leq \frac{- \nabla f(\alpha^k){j^2(k)}}{y{j^2(k)}} \leq ... \leq \frac{- \nabla f(\alpha^k){j^{q_2}(k)}}{y{j^{q_2}(k)}}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bj%5E1(k)%7D%7D%7By%7Bj%5E1(k)%7D%7D%20%5Cleq%20%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bj%5E2(k)%7D%7D%7By%7Bj%5E2(k)%7D%7D%20%5Cleq%20...%20%5Cleq%20%5Cfrac%7B-%20%5Cnabla%20f(%5Calpha%5Ek)%7Bj%5E%7Bq_2%7D(k)%7D%7D%7By%7Bj%5E%7Bq_2%7D(k)%7D%7D">

The ideal stopping criteria is satisfaction of the KKT conditions shown bellow. 

<!-- $m(\alpha^k) = max_{i \in R(\alpha^k)} \{ - \nabla f(\alpha^k)i y_i \} \leq min{j \in S(\alpha^k)} \{ - \nabla f(\alpha^k)_j y_j \} = M(\alpha^k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=m(%5Calpha%5Ek)%20%3D%20max_%7Bi%20%5Cin%20R(%5Calpha%5Ek)%7D%20%5C%7B%20-%20%5Cnabla%20f(%5Calpha%5Ek)i%20y_i%20%5C%7D%20%5Cleq%20min%7Bj%20%5Cin%20S(%5Calpha%5Ek)%7D%20%5C%7B%20-%20%5Cnabla%20f(%5Calpha%5Ek)_j%20y_j%20%5C%7D%20%3D%20M(%5Calpha%5Ek)">

But generally it takes a lot of iterations to satisfy this inequality exactly and this extremely increases the computation time. To avoid this, we let the violation of the KKT condition up to some small threshold which is our tolerance hyper-parameter equal to `0.0001`

### Q3

![Untitled](SVM_from_scratch%20c7258e6a31c84f9a883814224f083139/Untitled%201.png)

Here we have implemented the Sequential Minimal Optimization [(SMO)](https://en.wikipedia.org/wiki/Sequential_minimal_optimization) algorithm to solve the same soft margin svm quadratic problem with an iterative method. SMO is a decomposition method. The value of q is fixed to 2, all the time. Fixing q=2 makes the sub problem a convex quadratic problem of two variables which can be solved analytically. In the selection of these 2 indices, we use the Most Violating (MVP) pairs. Two indices <!-- $i \in I(\alpha)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=i%20%5Cin%20I(%5Calpha)"> and <!-- $j \in J(\alpha)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=j%20%5Cin%20J(%5Calpha)"> are MVP pairs, if and only if equations bellow are satisfied and KKT condition not satisfied.

<!-- $I(\alpha) = \{i : i \in argmax_{i \in R(\alpha)} \{ - \nabla f(\alpha)_i y_i \}  \}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=I(%5Calpha)%20%3D%20%5C%7Bi%20%3A%20i%20%5Cin%20argmax_%7Bi%20%5Cin%20R(%5Calpha)%7D%20%5C%7B%20-%20%5Cnabla%20f(%5Calpha)_i%20y_i%20%5C%7D%20%20%5C%7D">

<!-- $J(\alpha) = \{j : j \in argmin_{j \in S(\alpha)} \{ - \nabla f(\alpha)_j y_j \} \}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=J(%5Calpha)%20%3D%20%5C%7Bj%20%3A%20j%20%5Cin%20argmin_%7Bj%20%5Cin%20S(%5Calpha)%7D%20%5C%7B%20-%20%5Cnabla%20f(%5Calpha)_j%20y_j%20%5C%7D%20%5C%7D">