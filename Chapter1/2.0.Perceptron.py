# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# four samples with two inputs each
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

# correct output/class for each sample
correct_outputs = [False, False, False, True]
outputs = [] # container to store the output

# Set weight1, weight2, and bias
# correct weight to assign each sample to the correct labels/class
weight1 = 2.0
weight2 = 2.0
bias = -4.0

# Generate score and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    print(f"input 1 & 2: {(test_input[0], test_input[1])}; linear_combination: {linear_combination};  predicted output: {output};  is_correct_string: {is_correct_string}") 


# %% [markdown]
#

# %% [markdown]
# # Implementation of Perceptron Algroithm to solve Logical AND Problem

# %%
import plotly.graph_objects as go #type: ignore
import numpy as np #type: ignore

# %% [markdown]
#

# %%
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = [] # container to store the output

# Set weight1, weight2, and bias
weight1 = 2.0
weight2 = 2.0
bias = -4.0

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    print(f"input 1 & 2: {(test_input[0], test_input[1])}; linear_combination: {linear_combination};  predicted output: {output};  is_correct_string: {is_correct_string}") 


# %%
# Augment the input vector with bias = 1 as the first element in the vector.
# two inputs and one bias as input = three inputs in total 
inputs = [(1,0, 0), (1,0, 1), (1,1, 0), (1,1, 1)]
correct_outputs = [False, False, False, True]

# convert each sample to numpy array
input_train = [np.array(item) for item in inputs] # shape = 4 x 3
# convert boolean outputs into their integer equivalent
labels = [int(item) for item in correct_outputs]

# extract the input (x1) and input (x2) from the input array
x1 = np.array([item[1] for item in input_train])
x2 = np.array([item[2] for item in input_train])


# %%
# Plot the data points
fig = go.Figure()
# fig.add_trace(go.Scatter(x=x1, y=x2, fill='tozeroy', line_color = 'red', mode='lines'))
fig.add_trace(go.Scatter(x = x1[:3], y =x2[:3], mode ='markers',
                    marker = dict(color ='black', size = 20), name = 'False'))
fig.add_trace(go.Scatter(x = x1[3:], y =x2[3:], mode ='markers',
                    marker = dict(color ='green', size = 20), name = 'True'))
fig.update_layout(xaxis_title = "x1", yaxis_title = "x2", title="Boolean Logic for the AND Problem",
                legend_title = "True/False Data Points")
fig.show()

# %% [markdown]
# # Equation of a Line

# %% [markdown]
# # Equation of a Line
#
# **Plot the line**
# $$x_0 \times w_0 + x_1 \times w_1 + x_2 \times w_2 = 0 $$
# $$x_2 = - \frac {\left(x_0 \times w_0 + x_1 \times w_1 \right)}{w_2} $$
# where $x_2 = y, x_1 = x, x_0 = intercept$

# %%
# find the value of weights that will solve the problem
# Step 1: Pick random weights with the bias weight the first element
np.random.seed(7) # a number to make sure that we get the same random numbers everytime
weights = np.random.randn(3) 
init_weights = weights.copy()
print(f"W0: {weights[0]}, W1: {weights[1]}, W2: {weights[2]}")

# %%
# Draw the first boundary line using the random weights
x_pred = (-1 * (1 * init_weights[0].item() + x1 * init_weights[1].item()))/ init_weights[2].item()
fig = go.Figure()
# fig.add_trace(go.Scatter(x=x1, y=x2, fill='tozeroy', line_color = 'red', mode='lines'))
fig.add_trace(go.Scatter(x = x1[:3], y =x2[:3], mode ='markers',
                    marker = dict(color ='black', size = 20), name = '- data point'))
fig.add_trace(go.Scatter(x = x1[3:], y =x2[3:], mode ='markers',
                    marker = dict(color ='green', size = 20), name = '+ data point'))
fig.add_trace(go.Scatter(x=x1, y=x_pred, line_color = 'red', mode='lines',
                             line = dict(width = 5),   name = '1st predicted line'))
fig.update_layout(xaxis_title = "x1", yaxis_title = "x2", title="Boolean Logic for the AND Problem",
                legend_title = "True/False Data Points & Boundary Line")
fig.show()


# %%
# step 2

# Compute the core

def compute_score(X, y, w, lr=0.4,epoch = 8):
    '''
    lr = learning rate
    epoch = number of iterations
    '''
    outputs = []
    for i in range(epoch):
        for ind in range(len(X)):
            score = np.dot(w, X[ind]) 
            y_hat = int(score >= 0 )
            if y_hat - y[ind] == 1:
                w -= lr * X[ind]
            elif y_hat - y[ind] == -1:
                w += lr * X[ind]
        # save for plotting
        x_pred = (-1 * (1 * w[0].item() + x1 * w[1].item()))/ w[2].item()
        outputs.append(x_pred)    
    return w, outputs

w, outputs = compute_score(input_train, labels, weights)

# Output
# w = array([-1.1094743 ,  0.73406263,  0.43282016])


# %%
# draw the last boundary line using the new weights from above
x_pred = (-1 * (1 * -1.1094743 + x1 * 0.73406263))/ 0.43282016
fig = go.Figure()
# fig.add_trace(go.Scatter(x=x1, y=x2, fill='tozeroy', line_color = 'red', mode='lines'))
fig.add_trace(go.Scatter(x = x1[:3], y =x2[:3], mode ='markers',
                    marker = dict(color ='black', size = 20), name = '- data point'))
fig.add_trace(go.Scatter(x = x1[3:], y =x2[3:], mode ='markers',
                    marker = dict(color ='green', size = 20), name = '+ data point'))
fig.add_trace(go.Scatter(x=x1, y=x_pred, line_color = 'black', mode='lines',
                             line = dict(width = 5),   name = 'last predicted line'))
fig.update_layout(xaxis_title = "x1", yaxis_title = "x2", title="Boolean Logic for the AND Problem",
                legend_title = "True/False Data Points & Boundary Line")

fig.show()

# %%
len(outputs)
outputs

# %%
fig = go.Figure()
colors = ['red', 'green', 'blue', 'gray', 'yellow', 'purple', 'violet', 'black']
fig.add_trace(go.Scatter(x = x1[:3], y =x2[:3], mode ='markers',
                    marker = dict(color ='black', size = 20), name = 'False'))
fig.add_trace(go.Scatter(x = x1[3:], y =x2[3:], mode ='markers',
                    marker = dict(color ='green', size = 20), name = 'True'))
for ind in range(len(outputs)):
    if ind == 0:
        fig.add_trace(go.Scatter(x=x1, y=outputs[ind], line_color = 'red', mode='lines',
                                line = dict(width = 5), name = 'first line'))
    elif ind == 7:
        fig.add_trace(go.Scatter(x=x1, y=outputs[ind], line_color = 'black', mode='lines',
                                line = dict(width = 5), name = 'last line'))
    else:
        fig.add_trace(go.Scatter(x=x1, y=outputs[ind], line_color = colors[ind], mode='lines',
                                line = dict(width = 2), name = f'line{ind}'))
fig.update_layout(xaxis_title = "x1", yaxis_title = "x2", title="Implementation of the Perceptron Algorithm for the Logical AND Problem",
                legend_title = "True/False Data Points & Boundary Lines")
fig.update_yaxes(tickvals=[-25, -20, -15, -10, -5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0])

fig.show()

# %% [markdown]
# # Perceptron Trick with One Example 

# %% [markdown]
# Imagine a boundary line of the form:
#
# $$ 3x_1 + 4x_2 - 10 =0$$
#
# And a misclassified blue point at (1,1). How will you adjust the parameters to correctly  classify this point.

# %%
x1 = np.linspace(0, 3)
x2 = (-3 * x1 + 10)/4
fig = go.Figure()
# fig.add_trace(go.Scatter(x=x1, y=x2, fill='tozeroy', line_color = 'red', mode='lines'))
fig.add_trace(go.Scatter(x = [1], y =[1], mode ='markers',
                    marker = dict(color ='blue', size = 20), name = '+1 data point'))
fig.add_trace(go.Scatter(x=x1, y=x2, fill='tozeroy', line_color = 'red', mode='lines',
                             line = dict(width = 5),   name = 'negative data point region'))
fig.show()


# %%
w1 = 3; w2 = 4; bias = -10
learn_rate = 0.1
fig = go.Figure()
fig.add_trace(go.Scatter(x=x1, y=x2, line_color = 'red', mode='lines',
                               line = dict(width = 5), name = 'negative data point region'))
for i in range(1,14):
    if i== 13:
        w1 += learn_rate; w2 += learn_rate; bias += learn_rate
        x2 = (-( w1) * x1 + (-bias))/(w2)
        fig.add_trace(go.Scatter(x=x1, y=x2, line_color = 'black', mode='lines',
                line = dict(width = 5)))
    else:
        w1 = w1 + learn_rate; w2 = w2 + learn_rate; bias = bias + learn_rate

        x2 = (-( w1) * x1 + (-bias))/(w2)
        fig.add_trace(go.Scatter(x=x1, y=x2, line_color = 'gray', mode='lines'))
fig.add_trace(go.Scatter(x = [1], y =[1], mode ='markers',
                    marker = dict(color ='blue', size = 20), name = '+1 data point'))


fig.show()

# %%
