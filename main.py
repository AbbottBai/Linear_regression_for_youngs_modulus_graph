import math
import matplotlib.pyplot as plt
import numpy as np
import time

x_train = [0, 0.0004, 0.00053333, 0.00126666, 0.00193333, 0.004, 0.0104]
y_train = [0, 38506876.23, 77013752.46, 115520628.7, 154027504.9, 192534381.1, 231041257.4]

w = [1,1]
b = 1

plt.scatter(x_train, y_train, color='red', label='Training Points')
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.title('Training values (CLOSE TAB TO CONTINUE)')
plt.legend()
plt.show()

def standardisation(x_train, y_train):
   x_squared_sum = 0
   for i in range(len(x_train)):
       x_squared_sum += x_train[i] ** 2

   avg_x = sum(x_train) / len(x_train)
   x_standard_deviation = math.sqrt((x_squared_sum / len(x_train))
                                  - (sum(x_train) / len(x_train)) ** 2)
   for i in range(len(x_train)):
       x_train[i] = (x_train[i] - avg_x) / x_standard_deviation

   y_squared_sum = 0
   for i in range(len(y_train)):
       y_squared_sum += y_train[i] ** 2

   avg_y = sum(y_train) / len(y_train)
   y_standard_deviation = math.sqrt((y_squared_sum / len(y_train))
                                  - (sum(y_train) / len(y_train)) ** 2)
   for i in range(len(y_train)):
       y_train[i] = (y_train[i] - avg_y) / y_standard_deviation

   return x_train, y_train, avg_x, x_standard_deviation, avg_y, y_standard_deviation

x_train, y_train, avg_x, x_standard_deviation, avg_y, y_standard_deviation = standardisation(x_train, y_train)


def model(w_in, b_in, x_train):
   x_out = []
   for i in range(len(x_train)):
       current = (w_in[0] * x_train[i]
                  + w_in[1] * x_train[i]**2
                  + b_in)
       x_out.append(current)

   return x_out

"""y_hat = model(w,b,x_train,y_train)
print(y_hat)"""

def loss_func(w_in, b_in, x_train, y_train):
   loss = 0
   y_hat = model(w_in, b_in, x_train)
   for i in range(len(y_hat)):
       loss += (y_hat[i] - y_train[i])**2
   loss = loss / (2*len(y_hat))
   return loss, y_hat

def gradient_descent(w_in, b_in, x_train, y_train):
   alpha = 0.01
   l = 0.001

   for i in range(100000):
       loss, y_hat = loss_func(w_in, b_in, x_train, y_train)
       print(f"iteration {i}, loss = {loss}")

       total_loss_w0 = 0
       for j in range(len(y_hat)):
           total_loss_w0 += (y_hat[j] - y_train[j]) * x_train[j]
       avg_loss_w0 = total_loss_w0 / len(y_hat)

       total_loss_w1 = 0
       for j in range(len(y_hat)):
           total_loss_w1 += (y_hat[j] - y_train[j]) * x_train[j]**2
       avg_loss_w1 = total_loss_w1 / len(y_hat)

       total_loss_b = 0
       for j in range(len(y_hat)):
           total_loss_b += (y_hat[j] - y_train[j])
       avg_loss_b = total_loss_b / len(y_hat)

       w_in[0] = w_in[0] - alpha * avg_loss_w0 - (l/len(y_hat)) * w_in[0]
       w_in[1] = w_in[1] - alpha * avg_loss_w1 - (l/len(y_hat)) * w_in[1]
       b_in = b_in - alpha * avg_loss_b

   return w_in, b_in

w_out, b_out = gradient_descent(w, b, x_train, y_train)
print(f"\nfinal w: {w_out}")
print(f"\nfinal b: {b_out}")

print("\nIMPORTANT: the values of the graph were normalized so this new function only works for the normalized data")
print(f"The NORMALIZED equation of the graph is {w_out[1]}X^2 + {w_out[0]}X + {b_out}")

def plot_results(w_out, b_out, x_train, y_train):
   x_vals_smooth = np.linspace(min(x_train), max(x_train), 100)
   y_vals_final = model(w_out, b_out, x_vals_smooth)

   plt.scatter(x_train, y_train, color='red', label='Training Points')
   plt.plot(x_vals_smooth, y_vals_final, color='blue', label='Fitted Quadratic')
   plt.xlabel('Strain')
   plt.ylabel('Stress')
   plt.title('Young Modulus Practical (CLOSE TAB TO CONTINUE)')
   plt.legend()
   plt.show()

plot_results(w_out, b_out, x_train, y_train)

def predict(w, b, avg_x, x_standard_deviation, avg_y, y_standard_deviation):
   print("\nplease enter a strain value for the corresponding gradient and stress")
   strain = float(input(">"))
   strain_copy = strain
   strain = (strain - avg_x) / x_standard_deviation
   coefficient = 2 * w[1]
   time.sleep(0.3)
   print("\nIMPORTANT: the values of the graph were normalized so this new function only works for the normalized data")
   time.sleep(0.3)
   print(f"The NORMALIZED equation for the tangent (derivative of function) is {coefficient}X + {w[0]}")

   gradient = 2 * w[1] * strain + w[0]
   time.sleep(0.5)
   print(f"\nIMPORTANT: the gradient under this line of text is calculated from the derivative of the function")
   time.sleep(0.3)
   print(f"The gradient of the tangent when strain is {strain_copy} is {round(gradient, 3)} (rounded to 3 d.p.)")

   stress = w[1] * (strain ** 2) + w[0] * strain + b
   deregularized_stress = y_standard_deviation * stress + avg_y

   secondary_gradient = deregularized_stress / strain_copy
   time.sleep(0.5)
   print(f"\nIMPORTANT: the gradient under this line of text is calculated from stress over strain")
   time.sleep(0.3)
   print(f"The second gradient of the tangent when strain is {strain_copy} is {secondary_gradient}")

   time.sleep(0.3)
   print(f"\nThe corresponding stress when strain equals {strain_copy} is {deregularized_stress} Nm^-2")


predict(w_out, b_out, avg_x, x_standard_deviation, avg_y, y_standard_deviation)