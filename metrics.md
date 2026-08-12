Consider $f$ to be the right hand side of an ODE with $d$ species. We train a neural ode that can be represented as $\hat{f}_{\theta}$. To train the neural ode, we use a temporal discretization with $N$ points. 


Relative trajectory error per species. 

$$E_{traj, j} = \frac{\sum_{i=1}^N |\hat{y}_j(t_i) - y_j(t_i)|^2}{\sum_{i=1}^N |y_j(t_i)|^2}$$

Aggregated relative trajectory error. 
$$E_{traj,j} =\frac1d\sum_{j=1}^d E_{traj, j}$$

Velocity field error. 
$$E_{VF, j} = \frac{\sum_{i=1}^N |\hat{f}_{\theta,j}(y(t_i) - f_{j}(y(t_i)|^2}{\sum_{i=1}^N |f_{j}(y(t_i)|^2}$$

Aggregated velocity field error. 
$$E_{VF} =\frac1d\sum_{j=1}^d E_{VF, j}$$

Spectral error. 
$$E_{spec, j} = \frac{\sum_{i=1}^N |\hat{\lambda}_k(t_i) - \lambda_j(t_i)|^2}{\sum_{i=1}^N |\lambda_j(t_i)|^2}$$

Aggregated spectral error. 
$$E_{spec} =\frac1d\sum_{j=1}^d E_{spec, k}$$


Things to note. 
- For eigenvalues use the by $\lambda$ and $\hat{\lambda}$ I refer to the absolute value of the (potentially) complex number. 
- I want to save the aggregated metrics, as well as each species. the each species errors should be saved in a list. 
- Save all this metrics both on the training set, and the extrapolated time set. 
- 