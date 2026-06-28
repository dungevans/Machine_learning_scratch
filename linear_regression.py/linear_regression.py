import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/home/dung/machinelearning/linear_regression.py/score.csv') 
def loss ( m, b  , points ) : 
    total_error = 0  
    for i in range ( len ( points )) : 
        x = points.iloc[i].studytime
        y =  points.iloc[i].score
        total_error = ((m*x + b )- y)**2
    total_error = total_error/len ( points )


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    
    m = m_now - (L * m_gradient)
    b = b_now - (L * b_gradient)
    return m, b

m = 1
b = 2
L = 0.0001  
epochs = 300 


for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

print(f"Kết quả: m = {m}, b = {b}")


plt.scatter(data.studytime, data.score, color="black")

x_range = [min(data.studytime), max(data.studytime)]
plt.plot(x_range, [m * x + b for x in x_range], color="red")
plt.show () 